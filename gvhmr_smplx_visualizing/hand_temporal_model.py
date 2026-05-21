"""
Hand Temporal Model v4 Final
────────────────────────────────────────────────
변경 사항 (v4 revised 대비):
- mask_start/mask_len → mask (T, 1) 배열로 변경
  : blink 여러 개, spike 구간 전부 정확히 반영
- spike 구간도 mask에 포함
- 추론 초반 None 처리: 첫 유효 포즈 먼저 찾기
목표:
- 손가락이 갑자기 펴지는 현상 줄이기
- 1~3프레임 spike 제거
- 8~15프레임 이어지는 실패 패턴 완화
- 전체적으로 손가락 pose 변화량 부드럽게 만들기
"""

import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from smplx import MANO


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 (Model Architecture)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class HandTemporalModel(nn.Module):
    def __init__(self, pose_dim=45, hidden=256, n_heads=4, n_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(pose_dim, hidden)
        self.pos_enc    = SinusoidalPE(hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.local_conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.conv_norm  = nn.LayerNorm(hidden)

        self.output_proj = nn.Linear(hidden, pose_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        identity = x

        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)

        x = x.transpose(1, 2)
        x = self.local_conv(x)
        x = torch.nn.functional.gelu(x)
        x = x.transpose(1, 2)
        x = self.conv_norm(x)

        delta = self.output_proj(x)
        return identity + delta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Grip Prior Loss (프레임별 반환)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def grip_prior_loss_per_frame(pred_joints, handle_radius=0.015):
    n_joints = pred_joints.shape[2]

    if n_joints >= 21:
        fingertips  = pred_joints[:, :, [4,8,12,16,20], :]
        palm_center = pred_joints[:, :, [1,5,9,13,17], :].mean(dim=2)
        mid_mcp     = pred_joints[:, :, 9, :]
    else:
        fingertips  = pred_joints[:, :, [3,6,9,12,15], :]
        palm_center = pred_joints[:, :, [1,4,7,10,13], :].mean(dim=2)
        mid_mcp     = pred_joints[:, :, 7, :]

    wrist       = pred_joints[:, :, 0, :]
    handle_axis = mid_mcp - wrist
    handle_axis = handle_axis / (torch.norm(handle_axis, dim=-1, keepdim=True) + 1e-6)

    palm_center_exp = palm_center.unsqueeze(2)
    tip_vec         = fingertips - palm_center_exp

    axis_exp  = handle_axis.unsqueeze(2)
    proj      = (tip_vec * axis_exp).sum(dim=-1, keepdim=True) * axis_exp
    perp_dist = torch.norm(tip_vec - proj, dim=-1)

    loss_per_finger = torch.relu(perp_dist - handle_radius)
    loss_per_frame  = loss_per_finger.mean(dim=-1, keepdim=True)

    return loss_per_frame
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터셋 (Dataset)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HandPoseDataset(Dataset):
    def __init__(self, npz_paths, seq_len=31,
                 noise_std=0.10, spike_std=0.3,
                 is_train=True, max_mask_ratio=0.2,
                 blink_prob=0.7):
        if isinstance(npz_paths, str):
            npz_paths = [npz_paths]

        self.seq_len            = seq_len
        self.noise_std          = noise_std
        self.spike_std          = spike_std
        self.is_train           = is_train
        self.max_mask_ratio     = max_mask_ratio
        self.blink_prob         = blink_prob
        self.current_mask_ratio = 0.0
        self.samples            = []

        for path in npz_paths:
            data        = np.load(path)
            poses       = data['hand_pose'].astype(np.float32)
            seq_lengths = data['seq_lengths']

            start = 0
            for L in seq_lengths:
                end = start + L
                for i in range(start, end - seq_len + 1):
                    self.samples.append(poses[i: i + seq_len])
                start = end

        print(f"  샘플 수: {len(self.samples):,}")

    def set_epoch(self, epoch, total_epochs):
        self.current_mask_ratio = ((epoch + 1) / total_epochs) * self.max_mask_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gt    = self.samples[idx].copy()
        noisy = gt.copy()
        rng   = np.random.default_rng() if self.is_train \
                else np.random.default_rng(seed=idx)

        # (T, 1) mask 배열: 이상치 구간 전부 표시
        mask = np.zeros((self.seq_len, 1), dtype=np.float32)

        # 1. jitter
        noisy += rng.normal(0, self.noise_std, noisy.shape).astype(np.float32)

        # 2. spike + mask 표시
        n_spike   = rng.integers(1, 4)
        spike_idx = rng.choice(self.seq_len, size=int(n_spike), replace=False)
        for si in spike_idx:
            noisy[si] += rng.normal(0, self.spike_std, 45).astype(np.float32)
        mask[spike_idx] = 1.0  # spike 구간 mask 표시

        # 3. 깜빡임 패턴 4가지 + mask 표시
        if rng.random() < self.blink_prob:
            blink_type = rng.integers(0, 4)
            n_blink    = rng.integers(1, 4)

            for _ in range(n_blink):
                if blink_type == 0:
                    # 서서히 펴지는 패턴 (8~15프레임)
                    blink_len   = int(rng.integers(8, 15))
                    blink_start = int(rng.integers(0, max(1, self.seq_len - blink_len)))
                    for j in range(blink_len):
                        ratio = np.sin(np.pi * j / blink_len)
                        noisy[blink_start + j] *= (1.0 - ratio * 0.9)

                elif blink_type == 1:
                    # 갑자기 완전히 펴짐 (1~3프레임)
                    blink_len   = int(rng.integers(1, 4))
                    blink_start = int(rng.integers(0, max(1, self.seq_len - blink_len)))
                    noisy[blink_start: blink_start + blink_len] = 0.0

                elif blink_type == 2:
                    # 손가락 하나만 펴짐
                    blink_len     = int(rng.integers(3, 8))
                    blink_start   = int(rng.integers(0, max(1, self.seq_len - blink_len)))
                    finger        = rng.integers(0, 5)
                    finger_joints = [finger*3, finger*3+1, finger*3+2]
                    for j in range(blink_len):
                        ratio = np.sin(np.pi * j / blink_len)
                        for fj in finger_joints:
                            noisy[blink_start + j, fj] *= (1.0 - ratio * 0.9)

                elif blink_type == 3:
                    # 완전히 다른 모양
                    blink_len   = int(rng.integers(3, 8))
                    blink_start = int(rng.integers(0, max(1, self.seq_len - blink_len)))
                    random_pose = rng.normal(0, 0.5, (blink_len, 45)).astype(np.float32)
                    noisy[blink_start: blink_start + blink_len] = random_pose

                # blink 구간 mask 표시 (여러 blink 전부 반영)
                mask[blink_start: blink_start + blink_len] = 1.0

        # 4. 연속 마스킹 (curriculum)
        max_mask_len = int(self.seq_len * self.current_mask_ratio)
        if max_mask_len >= 1:
            cur_mask_len   = int(rng.integers(1, max_mask_len + 1))
            cur_mask_start = int(rng.integers(0, self.seq_len - cur_mask_len + 1))
            if rng.random() < 0.5:
                noisy[cur_mask_start: cur_mask_start + cur_mask_len] = 0.0
            else:
                random_pose = rng.normal(0, 0.5, (cur_mask_len, 45)).astype(np.float32)
                noisy[cur_mask_start: cur_mask_start + cur_mask_len] = random_pose
            mask[cur_mask_start: cur_mask_start + cur_mask_len] = 1.0

        return {
            "input": torch.from_numpy(noisy),
            "gt"   : torch.from_numpy(gt),
            "mask" : torch.from_numpy(mask),   # (T, 1)
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 손실 함수 (Loss Computation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_loss(pred, gt, inp, mask, mano_layer, device,
                 w_mask=5.0, w_accel=1.0, w_vel=0.1, w_joint3d=3.0,
                 w_grip_prior=0.1, w_preserve=0.2, handle_radius=0.015):
    """
    mask: (B, T, 1) - 이상치 구간 1.0, 정상 구간 0.0
    """
    B, T, _ = pred.shape

    grip_weight  = mask.to(device)                      # (B, T, 1) 이상치 구간
    weight       = 1.0 + grip_weight * (w_mask - 1.0)  # 이상치 구간에 w_mask 가중치
    normal_weight = 1.0 - grip_weight                   # 정상 구간

    # 1. Pose MSE Loss
    loss_pose = (((pred - gt) ** 2) * weight).mean()

    # 2. MANO 3D 관절 계산
    pred_flat     = pred.reshape(B * T, 45).float()
    gt_flat       = gt.reshape(B * T, 45).float()
    global_orient = torch.zeros(B * T, 3, device=device, dtype=torch.float32)
    betas         = torch.zeros(B * T, 10, device=device, dtype=torch.float32)

    with torch.amp.autocast("cuda", enabled=False):
        pred_output = mano_layer(betas=betas, global_orient=global_orient, hand_pose=pred_flat)
        gt_output   = mano_layer(betas=betas, global_orient=global_orient, hand_pose=gt_flat)

    pred_joints = pred_output.joints.reshape(B, T, -1, 3)
    gt_joints   = gt_output.joints.reshape(B, T, -1, 3)
    weight_3d   = weight.unsqueeze(-1)

    # 3. 3D 관절 L1 Loss
    loss_joint3d = ((pred_joints - gt_joints).abs() * weight_3d).mean()

    # 4. 속도 L1 Loss
    pred_vel = pred_joints[:, 1:] - pred_joints[:, :-1]
    gt_vel   = gt_joints[:, 1:] - gt_joints[:, :-1]
    loss_vel = torch.mean(torch.abs(pred_vel - gt_vel))

    # 5. 가속도 L1 Loss
    pred_accel = pred_joints[:, 2:] - 2 * pred_joints[:, 1:-1] + pred_joints[:, :-2]
    gt_accel   = gt_joints[:, 2:] - 2 * gt_joints[:, 1:-1] + gt_joints[:, :-2]
    loss_accel = torch.mean(torch.abs(pred_accel - gt_accel))

    # 6. Grip Prior Loss (이상치 구간에만 적용)
    loss_gp_frame   = grip_prior_loss_per_frame(pred_joints, handle_radius)
    loss_grip_prior = (loss_gp_frame * grip_weight).sum() \
                    / (grip_weight.sum() + 1e-6)

    # 7. Input Preservation Loss (정상 구간에서 입력 보존)
    loss_preserve = (((pred - inp) ** 2) * normal_weight).sum() \
                  / (normal_weight.sum() * pred.shape[-1] + 1e-6)

    total = (loss_pose
             + w_joint3d    * loss_joint3d
             + w_accel      * loss_accel
             + w_vel        * loss_vel
             + w_grip_prior * loss_grip_prior
             + w_preserve   * loss_preserve)

    return total, loss_pose, loss_accel, loss_vel, loss_joint3d, loss_grip_prior, loss_preserve


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 학습 루프 (Training Pipeline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train(
    train_paths,
    val_path,
    save_path      = "hand_temporal_model.pth",
    seq_len        = 31,
    epochs         = 50,
    batch_size     = 64,
    lr             = 3e-4,
    max_mask_ratio = 0.2,
    blink_prob     = 0.7,
    w_mask         = 5.0,
    w_accel        = 1.0,
    w_vel          = 0.1,
    w_joint3d      = 3.0,
    w_grip_prior   = 0.1,
    w_preserve     = 0.2,
    handle_radius  = 0.015,
    mano_path      = "mano",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    print("\n[train]")
    train_dataset = HandPoseDataset(
        train_paths, seq_len=seq_len, is_train=True,
        max_mask_ratio=max_mask_ratio, blink_prob=blink_prob
    )
    print("[val]")
    val_dataset = HandPoseDataset(
        val_path, seq_len=seq_len, is_train=False,
        max_mask_ratio=max_mask_ratio, blink_prob=blink_prob
    )
    val_dataset.set_epoch(epochs // 2, epochs)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    model      = HandTemporalModel().to(device)
    mano_layer = MANO(model_path=mano_path, is_rhand=True, use_pca=False).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp    = torch.cuda.is_available()
    scaler     = GradScaler("cuda", enabled=use_amp)

    best_val = float('inf')
    header = (f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  "
              f"{'pose':>8}  {'accel':>8}  {'vel':>8}  "
              f"{'j3d':>8}  {'grip':>8}  {'prsv':>8}  {'mask_r':>7}")
    print(f"\n{'─'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")

    for epoch in range(epochs):
        train_dataset.set_epoch(epoch, epochs)
        mask_ratio = train_dataset.current_mask_ratio

        # ── train ──────────────────────────────────────────
        model.train()
        total_train = 0.0

        for batch in train_loader:
            x    = batch["input"].to(device)
            gt   = batch["gt"].to(device)
            mask = batch["mask"].to(device)   # (B, T, 1)

            optimizer.zero_grad()

            with autocast("cuda", enabled=use_amp):
                pred = model(x)
                loss, _, _, _, _, _, _ = compute_loss(
                    pred, gt, x, mask,
                    mano_layer, device,
                    w_mask, w_accel, w_vel, w_joint3d,
                    w_grip_prior, w_preserve, handle_radius
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train += loss.item()

        # ── val ────────────────────────────────────────────
        model.eval()
        total_val = 0.0
        sum_lp = sum_la = sum_lv = sum_lj = sum_lg = sum_lprsv = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x    = batch["input"].to(device)
                gt   = batch["gt"].to(device)
                mask = batch["mask"].to(device)

                with autocast("cuda", enabled=use_amp):
                    pred = model(x)
                    val_loss, lp, la, lv, lj, lg, lprsv = compute_loss(
                        pred, gt, x, mask,
                        mano_layer, device,
                        w_mask, w_accel, w_vel, w_joint3d,
                        w_grip_prior, w_preserve, handle_radius
                    )

                total_val  += val_loss.item()
                sum_lp     += lp.item()
                sum_la     += la.item()
                sum_lv     += lv.item()
                sum_lj     += lj.item()
                sum_lg     += lg.item()
                sum_lprsv  += lprsv.item()

        scheduler.step()

        n         = len(val_loader)
        avg_train = total_train / len(train_loader)
        avg_val   = total_val   / n

        saved = ""
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), save_path)
            saved = "  ← 저장"

        print(f"{epoch+1:>6}  {avg_train:>10.5f}  {avg_val:>10.5f}  "
              f"{sum_lp/n:>8.5f}  {sum_la/n:>8.5f}  {sum_lv/n:>8.5f}  "
              f"{sum_lj/n:>8.5f}  {sum_lg/n:>8.5f}  {sum_lprsv/n:>8.5f}  "
              f"{mask_ratio:>7.3f}{saved}")

    print(f"\n학습 완료 | best val: {best_val:.6f} | 저장: {save_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 추론 (Inference)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TemporalSmoother:
    def __init__(self, model_path, seq_len=31, device=None):
        self.device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len

        self.model = HandTemporalModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def process_video(self, pose_list):
        half = self.seq_len // 2
        N    = len(pose_list)

        # 첫 유효 포즈 먼저 찾기
        first_valid = None
        for p in pose_list:
            if p is not None:
                first_valid = p.cpu().numpy().flatten().astype(np.float32)
                break
        last_valid = first_valid if first_valid is not None \
                     else np.zeros(45, dtype=np.float32)

        # None 처리: 이전 유효 포즈 사용
        poses = []
        for p in pose_list:
            if p is None:
                poses.append(last_valid.copy())
            else:
                arr        = p.cpu().numpy().flatten().astype(np.float32)
                poses.append(arr)
                last_valid = arr

        poses = np.stack(poses, axis=0)

        pad_start = np.repeat(poses[:1],  half, axis=0)
        pad_end   = np.repeat(poses[-1:], half, axis=0)
        padded    = np.concatenate([pad_start, poses, pad_end], axis=0)

        results = []
        for i in range(N):
            window = padded[i: i + self.seq_len]
            x      = torch.tensor(window).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.model(x)

            results.append(pred[0, half].unsqueeze(0))

        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 진입점 (Main Execution)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    train(
        train_paths    = [
            "ho3d_train.npz",
            "interhand_train.npz",
        ],
        val_path       = "interhand_val.npz",
        save_path      = "hand_temporal_model.pth",
        seq_len        = 31,
        epochs         = 50,
        batch_size     = 64,
        lr             = 3e-4,
        max_mask_ratio = 0.2,
        blink_prob     = 0.7,
        w_mask         = 5.0,
        w_accel        = 1.0,
        w_vel          = 0.1,
        w_joint3d      = 3.0,
        w_grip_prior   = 0.1,
        w_preserve     = 0.2,
        handle_radius  = 0.015,
        mano_path      = "mano",
    )