# -*- coding: utf-8 -*-
"""
HandPoseKalmanFilter
────────────────────────────────────────────────────────────
- 45차원 MANO hand_pose Kalman Filter
- HaMeR confidence 기반 관측 노이즈 조정
- 그립 스코어 기반 깜빡임 억제 (레퍼런스 포즈 불필요)
  → MANO joint 기하학으로 손가락 굴곡 자동 계산
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


# MANO joint 인덱스
_WRIST      = 0
_FINGERTIPS = [4, 8, 12, 16, 20]   # 엄지·검지·중지·약지·소지 끝
_MCPS       = [1, 5,  9, 13, 17]   # 각 손가락 MCP (너클)
_WRIST_16      = 0
_FINGERTIPS_16 = [3, 6, 9, 12]   # DIP 관절 (끝 마디): 검지·중지·약지·소지
_MCPS_16       = [1, 4, 7, 10]   # MCP 관절 (너클): 검지·중지·약지·소지


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 그립 스코어 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def grip_score_from_joints(joints: np.ndarray) -> float:
    """
    16 or 21 joint MANO 자동 감지 후 grip score 계산.
    """
    n = joints.shape[0]

    if n >= 21:
        # 21-joint: 실제 fingertip 사용
        tips  = joints[[4, 8, 12, 16, 20]]
        mcps  = joints[[1, 5,  9, 13, 17]]
        mid_mcp = joints[9]
    else:
        # 16-joint: DIP 관절을 fingertip 대용으로 사용
        tips  = joints[[3, 6, 9, 12]]
        mcps  = joints[[1, 4, 7, 10]]
        mid_mcp = joints[4]

    wrist       = joints[0]
    palm_center = mcps.mean(axis=0)
    hand_size   = np.linalg.norm(mid_mcp - wrist) + 1e-6

    tip_to_palm = np.linalg.norm(tips - palm_center, axis=1)
    closure     = 1.0 - np.clip(tip_to_palm / hand_size, 0.0, 1.0)

    # 21-joint이면 엄지 제외 (인덱스 0), 16-joint이면 전부 사용
    if n >= 21:
        return float(np.clip(closure[1:].mean(), 0.0, 1.0))
    else:
        return float(np.clip(closure.mean(), 0.0, 1.0))

def compute_all_grip_scores(
    all_poses    : np.ndarray,
    mano_path    : str   = "mano",
    smooth_sigma : float = 5.0,
) -> np.ndarray:
    from smplx import MANO

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mano_layer = MANO(model_path=mano_path, is_rhand=True,
                      use_pca=False).to(device)
    mano_layer.eval()

    N      = len(all_poses)
    scores = np.zeros(N, dtype=np.float32)

    with torch.no_grad():
        for i in range(N):
            pose = torch.tensor(
                all_poses[i:i+1], dtype=torch.float32
            ).to(device)                              # (1, 45)
            go   = torch.zeros(1, 3, device=device)

            out    = mano_layer(global_orient=go, hand_pose=pose)
            joints = out.joints[0].cpu().numpy()      # (21, 3)
            scores[i] = grip_score_from_joints(joints)

    smoothed = gaussian_filter1d(scores.astype(np.float64), sigma=smooth_sigma)
    return smoothed.astype(np.float32)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Kalman Filter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HandPoseKalmanFilter:
    """
    45차원 MANO hand_pose Kalman Filter.

    상태: [pose(45), velocity(45)] = 90차원
    관측: HaMeR hand_pose 45차원

    update()            : HaMeR confidence만 사용
    update_grip_aware() : HaMeR confidence + 그립 스코어 결합
                          (레퍼런스 포즈 불필요)
    """

    def __init__(
        self,
        pose_dim            = 45,
        fps                 = 60,
        process_var         = 1e-4,
        obs_var             = 1e-2,
        confidence_power    = 2.0,
        outlier_threshold   = 0.8,
        outlier_noise_scale = 10.0,
        # 그립 관련
        grip_ema_alpha      = 0.3,   # EMA 강도 (낮을수록 느리게 반응)
        blink_threshold     = 0.2,   # 이 이상 갑자기 떨어지면 깜빡임으로 판단
        grip_high           = 0.65,  # 그립 유지 기준
        release_low         = 0.35,  # 진짜 풀림 기준
        device              = None,
    ):
        self.pose_dim            = pose_dim
        self.state_dim           = pose_dim * 2
        self.dt                  = 1.0 / float(fps)
        self.process_var         = process_var
        self.obs_var             = obs_var
        self.confidence_power    = confidence_power
        self.outlier_threshold   = outlier_threshold
        self.outlier_noise_scale = outlier_noise_scale
        self.grip_alpha          = grip_ema_alpha
        self.blink_thresh        = blink_threshold
        self.grip_high           = grip_high
        self.release_low         = release_low

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.x = np.zeros((self.state_dim, 1), dtype=np.float32)
        self.P = np.eye(self.state_dim, dtype=np.float32)

        # 상태 전이 (등속 모델)
        self.A = np.eye(self.state_dim, dtype=np.float32)
        self.A[:pose_dim, pose_dim:] = np.eye(pose_dim) * self.dt

        # 관측 행렬
        self.H = np.zeros((pose_dim, self.state_dim), dtype=np.float32)
        self.H[:, :pose_dim] = np.eye(pose_dim)

        self.Q      = np.eye(self.state_dim, dtype=np.float32) * process_var
        self.base_R = np.eye(pose_dim, dtype=np.float32) * obs_var

        self.initialized   = False
        self.last_pose     = None
        self.smoothed_grip = 0.5   # EMA 초기값

    # ── 내부 유틸 ──────────────────────────────────────────────

    def reset(self):
        self.x             = np.zeros((self.state_dim, 1), dtype=np.float32)
        self.P             = np.eye(self.state_dim, dtype=np.float32)
        self.initialized   = False
        self.last_pose     = None
        self.smoothed_grip = 0.5

    def _to_numpy_pose(self, pose_tensor):
        if pose_tensor is None:
            return None
        if isinstance(pose_tensor, torch.Tensor):
            arr = pose_tensor.detach().cpu().numpy().reshape(-1)
        elif isinstance(pose_tensor, np.ndarray):
            arr = pose_tensor.reshape(-1)
        else:
            return None
        if arr.shape[0] < self.pose_dim:
            return None
        return arr[:self.pose_dim].astype(np.float32).reshape(self.pose_dim, 1)

    def _predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def _make_obs_R(self, confidence, z):
        """HaMeR confidence + outlier 정도로 관측 노이즈 R 계산."""
        conf    = float(np.clip(confidence, 0.05, 1.0))
        R_scale = 1.0 / (conf ** self.confidence_power)
        if self.last_pose is not None:
            jump = float(np.mean(np.abs(z - self.last_pose)))
            if jump > self.outlier_threshold:
                R_scale *= self.outlier_noise_scale
        return self.base_R * R_scale

    def _grip_confidence(self, raw_grip: float) -> float:
        """
        그립 스코어 EMA 평활화 후 confidence 계산.

        깜빡임(순간 하락) → confidence 낮춤 → 현재 프레임 무시
        진짜 풀림(점진적 하락) → confidence 높임 → 현재 프레임 반영
        """
        prev               = self.smoothed_grip
        self.smoothed_grip = (self.grip_alpha * raw_grip
                               + (1.0 - self.grip_alpha) * prev)
        drop = prev - raw_grip   # 얼마나 갑자기 떨어졌나

        # 그립 유지 중인데 갑자기 뚝 → 깜빡임
        if prev > self.grip_high and drop > self.blink_thresh:
            return 0.05

        # 점진적으로 낮아지는 중 → 진짜 풀림
        if self.smoothed_grip < self.release_low:
            return 0.9

        # 중간 상태
        return 0.2 + 0.7 * self.smoothed_grip

    def _kalman_update(self, z, R):
        y      = z - self.H @ self.x
        S      = self.H @ self.P @ self.H.T + R
        K      = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim, dtype=np.float32) - K @ self.H) @ self.P

    def _to_tensor(self):
        pose = self.x[:self.pose_dim].reshape(1, self.pose_dim)
        return torch.tensor(pose, dtype=torch.float32, device=self.device)

    # ── 공개 메서드 ────────────────────────────────────────────

    def update(self, pose_tensor, confidence=1.0):
        """HaMeR confidence만 사용하는 기본 업데이트."""
        z = self._to_numpy_pose(pose_tensor)

        if z is None:
            if not self.initialized:
                return None
            self._predict()
            return self._to_tensor()

        if not self.initialized:
            self.x[:self.pose_dim] = z
            self.last_pose         = z.copy()
            self.initialized       = True
            return self._to_tensor()

        self._predict()
        R = self._make_obs_R(confidence, z)
        self._kalman_update(z, R)
        self.last_pose = self.x[:self.pose_dim].copy()
        return self._to_tensor()

    def update_grip_aware(
        self,
        pose_tensor    ,
        confidence     : float = 1.0,
        raw_grip_score : float = 0.5,
    ):
        """
        HaMeR confidence + 그립 스코어를 결합한 업데이트.
        레퍼런스 포즈 없이 동작.

        Parameters
        ----------
        pose_tensor    : (1, 45) Tensor or None
        confidence     : HaMeR keypoint confidence (0~1)
        raw_grip_score : grip_score_from_joints() 로 계산한 값 (0~1)
                         미리 양방향 평활화된 값을 넣으면 더 안정적

        Returns
        -------
        (1, 45) Tensor
        """
        z = self._to_numpy_pose(pose_tensor)

        if z is None:
            if not self.initialized:
                return None
            self._predict()
            self._grip_confidence(raw_grip_score)   # EMA 업데이트
            return self._to_tensor()

        if not self.initialized:
            self.x[:self.pose_dim] = z
            self.last_pose         = z.copy()
            self.smoothed_grip     = raw_grip_score
            self.initialized       = True
            return self._to_tensor()

        self._predict()

        # HaMeR confidence × 그립 confidence → combined
        hamer_conf = float(np.clip(confidence, 0.05, 1.0))
        grip_conf  = self._grip_confidence(raw_grip_score)
        combined   = hamer_conf * grip_conf

        R = self._make_obs_R(combined, z)
        self._kalman_update(z, R)
        self.last_pose = self.x[:self.pose_dim].copy()
        return self._to_tensor()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# confidence 보정 유틸
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def refine_keypoint_confidence(
    raw_confidence,
    valid_count    = None,
    total_keypoints= 21,
):
    """
    HaMeR keypoint_confidiences → Kalman용 confidence 보정.
    valid_count가 있으면 valid_ratio도 반영.
    """
    conf = float(np.clip(raw_confidence, 0.05, 1.0))
    if valid_count is None:
        return conf
    valid_ratio = float(valid_count) / float(total_keypoints)
    return float(np.clip(conf * valid_ratio, 0.05, 1.0))