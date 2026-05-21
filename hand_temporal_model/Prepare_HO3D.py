"""
HO3D v3 전처리 파서
────────────────────────────────────────────────────────────────
출력: {split}_ho3d.npz
  - hand_pose   : (N, 45)  float32  axis-angle, global_orient 제외
  - seq_lengths : (S,)     int32    각 시퀀스의 프레임 수
  - mean        : (45,)    float32  정규화용 평균
  - std         : (45,)    float32  정규화용 표준편차

디렉토리 구조 가정:
  ho3d_root/
  ├── train/
  │   ├── ABF10/
  │   │   └── meta/
  │   │       ├── 0000.pkl
  │   │       └── ...
  │   └── ...
  └── val/
      └── ...
"""

import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm


# ── 유틸 ─────────────────────────────────────────────────────────

def load_pkl(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')


def flip_hand_pose(hand_pose: np.ndarray) -> np.ndarray:
    """
    오른손 axis-angle pose (45,) → 왼손으로 반전

    axis-angle r = θ·[nx, ny, nz] 에서
    YZ 평면 기준 반사(x 부호 반전)가 일반적이나,
    HO3D 카메라 좌표계(x-right, y-down, z-forward)에서는
    y·z 반전이 더 자주 쓰임.

    ※ 반드시 MANO forward pass + 시각화로 결과를 육안 확인할 것.
    """
    flipped = hand_pose.copy().reshape(15, 3)
    flipped[:, 1] *= -1   # y 반전
    flipped[:, 2] *= -1   # z 반전
    return flipped.reshape(45)


# ── 파서 ─────────────────────────────────────────────────────────

def parse_sequence(meta_dir: str, augment_left: bool):
    """
    하나의 시퀀스 디렉토리에서 pose 시퀀스를 파싱.

    Returns
    -------
    right_seq : np.ndarray (T, 45) or None
    left_seq  : np.ndarray (T, 45) or None  (augment_left=False 이면 None)
    """
    pkl_files = sorted(
        f for f in os.listdir(meta_dir) if f.endswith('.pkl')
    )
    if not pkl_files:
        return None, None

    right_frames, left_frames = [], []

    for fname in pkl_files:
        fpath = os.path.join(meta_dir, fname)
        try:
            data = load_pkl(fpath)

            # handPose: (48,) = global_orient(3) + finger_pose(45)
            hand_pose_full = data['handPose']           # (48,)
            hand_pose      = hand_pose_full[3:].astype(np.float32)  # (45,)

            right_frames.append(hand_pose)
            if augment_left:
                left_frames.append(flip_hand_pose(hand_pose))

        except Exception as e:
            print(f"  [WARN] {fname} 로드 실패: {e}")
            continue

    if not right_frames:
        return None, None

    right_seq = np.stack(right_frames, axis=0)                          # (T, 45)
    left_seq  = np.stack(left_frames,  axis=0) if augment_left else None
    return right_seq, left_seq


def prepare_ho3d(
    ho3d_root   : str,
    output_path : str,
    split       : str  = 'train',
    min_seq_len : int  = 31,
    augment_left: bool = True,
) -> None:
    """
    HO3D 전처리 메인 함수.

    Parameters
    ----------
    ho3d_root    : HO3D v3 루트 경로
    output_path  : 저장할 .npz 경로
    split        : 'train' | 'val'
    min_seq_len  : 이 값보다 짧은 시퀀스는 제외 (슬라이딩 윈도우 기준)
    augment_left : True면 오른손 반전으로 왼손 데이터도 생성
    """
    split_dir = os.path.join(ho3d_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"split 디렉토리 없음: {split_dir}")

    sequences = sorted(os.listdir(split_dir))

    right_seqs, left_seqs = [], []
    skipped = 0

    for seq_name in tqdm(sequences, desc=f"[{split}] 시퀀스 처리"):
        meta_dir = os.path.join(split_dir, seq_name, 'meta')
        if not os.path.isdir(meta_dir):
            continue

        right_seq, left_seq = parse_sequence(meta_dir, augment_left)

        if right_seq is None:
            skipped += 1
            continue

        # min_seq_len 미만 시퀀스 제외
        if len(right_seq) < min_seq_len:
            print(f"  [SKIP] {seq_name}: {len(right_seq)} 프레임 < {min_seq_len}")
            skipped += 1
            continue

        right_seqs.append(right_seq)
        if augment_left and left_seq is not None:
            left_seqs.append(left_seq)

    if not right_seqs:
        raise RuntimeError("유효한 시퀀스가 없습니다. 경로와 split을 확인하세요.")

    # 오른손 + 왼손 합치기
    all_seqs    = right_seqs + left_seqs
    seq_lengths = np.array([len(s) for s in all_seqs], dtype=np.int32)
    all_poses   = np.concatenate(all_seqs, axis=0).astype(np.float32)  # (N, 45)

    # 정규화 통계 (학습 split에서만 계산)
    mean = all_poses.mean(axis=0)   # (45,)
    std  = all_poses.std(axis=0)    # (45,)
    std  = np.where(std < 1e-6, 1.0, std)   # 분산 0인 차원 안전 처리

    np.savez(
        output_path,
        hand_pose   = all_poses,
        seq_lengths = seq_lengths,
        mean        = mean,
        std         = std,
    )

    # ── 요약 출력 ─────────────────────────────────────────────────
    n_right  = sum(len(s) for s in right_seqs)
    n_left   = sum(len(s) for s in left_seqs)

    print(f"\n{'='*50}")
    print(f"저장 완료 : {output_path}")
    print(f"{'='*50}")
    print(f"오른손 프레임  : {n_right:>8,}")
    print(f"왼손 프레임    : {n_left:>8,}")
    print(f"총 프레임      : {n_right + n_left:>8,}")
    print(f"총 시퀀스      : {len(all_seqs):>8,}")
    print(f"제외 시퀀스    : {skipped:>8,}")
    print(f"pose 평균 범위 : [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"pose std  범위 : [{std.min():.4f},  {std.max():.4f}]")
    print(f"{'='*50}\n")


# ── 검증 유틸 ─────────────────────────────────────────────────────

def verify_npz(npz_path: str, seq_len: int = 31) -> None:
    """
    저장된 npz 파일이 Dataset에서 제대로 쓰일 수 있는지 빠르게 검증.
    """
    data        = np.load(npz_path)
    hand_pose   = data['hand_pose']     # (N, 45)
    seq_lengths = data['seq_lengths']   # (S,)
    mean        = data['mean']          # (45,)
    std         = data['std']           # (45,)

    # 슬라이딩 윈도우 수 계산
    n_windows = sum(
        max(0, L - seq_len + 1) for L in seq_lengths
    )

    print(f"\n[검증] {npz_path}")
    print(f"  hand_pose shape  : {hand_pose.shape}")
    print(f"  시퀀스 수         : {len(seq_lengths)}")
    print(f"  프레임 수 합계    : {seq_lengths.sum():,}")
    print(f"  슬라이딩 윈도우   : {n_windows:,}  (seq_len={seq_len})")
    print(f"  mean shape       : {mean.shape}")
    print(f"  std  shape       : {std.shape}")
    print(f"  nan 포함 여부     : {np.isnan(hand_pose).any()}")
    print(f"  inf 포함 여부     : {np.isinf(hand_pose).any()}")


# ── 진입점 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HO3D v3 전처리 파서")
    parser.add_argument("--root",    type=str, default="C:/HO3D_v3")
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--seq_len", type=int, default=31,
                        help="최소 시퀀스 길이 (학습 seq_len과 동일하게)")
    parser.add_argument("--no_flip", action="store_true",
                        help="왼손 augmentation 비활성화")
    args = parser.parse_args()

    augment = not args.no_flip

    # train
    train_out = os.path.join(args.out_dir, "ho3d_train.npz")
    prepare_ho3d(
        ho3d_root    = args.root,
        output_path  = train_out,
        split        = 'train',
        min_seq_len  = args.seq_len,
        augment_left = augment,
    )
    verify_npz(train_out, seq_len=args.seq_len)

    # val
    val_out = os.path.join(args.out_dir, "ho3d_val.npz")
    prepare_ho3d(
        ho3d_root    = args.root,
        output_path  = val_out,
        split        = 'val',
        min_seq_len  = args.seq_len,
        augment_left = False,   # val은 augmentation 없이
    )
    verify_npz(val_out, seq_len=args.seq_len)