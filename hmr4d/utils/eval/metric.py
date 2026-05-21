from __future__ import annotations

from typing import Any

import numpy as np
import torch

from hmr4d.utils.eval.eval_utils import compute_jitter


BODY_JOINT_IDXS = tuple(range(22))
LEFT_HAND_JOINT_IDXS = tuple(range(25, 40))
RIGHT_HAND_JOINT_IDXS = tuple(range(40, 55))
FULL_BODY_JOINT_IDXS = BODY_JOINT_IDXS + LEFT_HAND_JOINT_IDXS + RIGHT_HAND_JOINT_IDXS
FOOT_VERTEX_IDXS = {
    "left": (3216, 3387),
    "right": (6617, 6787),
}
LEFT_ARM_IDXS = {"elbow": 18, "wrist": 20, "index_mcp": 25, "pinky_mcp": 34}
RIGHT_ARM_IDXS = {"elbow": 19, "wrist": 21, "index_mcp": 40, "pinky_mcp": 49}

SMPLX_PARAM_KEYS = (
    "betas",
    "body_pose",
    "global_orient",
    "transl",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "expression",
)


def _to_device_tensor_dict(smpl_params: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    filtered = {}
    for key in SMPLX_PARAM_KEYS:
        value = smpl_params.get(key)
        if torch.is_tensor(value):
            filtered[key] = value.to(device=device)
    return filtered


def _compute_accel_norm(joints: torch.Tensor, fps: float = 30.0) -> np.ndarray:
    if joints.shape[0] < 3:
        return np.zeros((0,), dtype=np.float32)
    accel = (joints[2:] - 2 * joints[1:-1] + joints[:-2]) * (fps**2)
    return torch.norm(accel, dim=-1).mean(dim=-1).cpu().numpy()


def _safe_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _normalize_vectors(vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norms = torch.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)
    return vectors / norms


def _compute_angle_deg(vec_a: torch.Tensor, vec_b: torch.Tensor) -> np.ndarray:
    vec_a = _normalize_vectors(vec_a)
    vec_b = _normalize_vectors(vec_b)
    cos = (vec_a * vec_b).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cos))
    return angles.cpu().numpy()


@torch.no_grad()
def build_smplx_joints(
    smpl_model: torch.nn.Module,
    smpl_params: dict[str, Any],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    device = torch.device(device)
    model_inputs = _to_device_tensor_dict(smpl_params, device)
    smpl_out = smpl_model(**model_inputs)
    joints = smpl_out.joints.detach().cpu()
    if joints.shape[1] < 55:
        raise ValueError(f"Expected at least 55 SMPL-X joints, got shape {tuple(joints.shape)}")
    return joints[:, :55]


@torch.no_grad()
def build_smplx_joints_and_verts(
    smpl_model: torch.nn.Module,
    smpl_params: dict[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    model_inputs = _to_device_tensor_dict(smpl_params, device)
    smpl_out = smpl_model(**model_inputs)
    joints = smpl_out.joints.detach().cpu()
    verts = smpl_out.vertices.detach().cpu()
    if joints.shape[1] < 55:
        raise ValueError(f"Expected at least 55 SMPL-X joints, got shape {tuple(joints.shape)}")
    return joints[:, :55], verts


def compute_temporal_metrics_from_joints(joints: torch.Tensor, fps: float = 30.0) -> dict[str, Any]:
    groups = {
        "fullbody": FULL_BODY_JOINT_IDXS,
        "body": BODY_JOINT_IDXS,
        "left_hand": LEFT_HAND_JOINT_IDXS,
        "right_hand": RIGHT_HAND_JOINT_IDXS,
    }

    metrics: dict[str, Any] = {}
    for name, idxs in groups.items():
        joint_group = joints[:, idxs, :]
        jitter = compute_jitter(joint_group, fps=fps)
        accel = _compute_accel_norm(joint_group, fps=fps)
        metrics[f"{name}_jitter"] = jitter
        metrics[f"{name}_accel"] = accel
        metrics[f"{name}_jitter_summary"] = _safe_summary(jitter)
        metrics[f"{name}_accel_summary"] = _safe_summary(accel)
    return metrics


def compute_forearm_palm_consistency(joints: torch.Tensor, fps: float = 30.0) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    arm_defs = {
        "left": LEFT_ARM_IDXS,
        "right": RIGHT_ARM_IDXS,
    }

    for side, idxs in arm_defs.items():
        elbow = joints[:, idxs["elbow"], :]
        wrist = joints[:, idxs["wrist"], :]
        index_mcp = joints[:, idxs["index_mcp"], :]
        pinky_mcp = joints[:, idxs["pinky_mcp"], :]

        forearm_vec = wrist - elbow
        palm_center = 0.5 * (index_mcp + pinky_mcp)
        palm_vec = palm_center - wrist

        angle_deg = _compute_angle_deg(forearm_vec, palm_vec)
        if angle_deg.shape[0] >= 2:
            angle_vel = np.abs(np.diff(angle_deg)) * fps
        else:
            angle_vel = np.zeros((0,), dtype=np.float32)

        metrics[f"{side}_forearm_palm_angle"] = angle_deg
        metrics[f"{side}_forearm_palm_angle_summary"] = _safe_summary(angle_deg)
        metrics[f"{side}_forearm_palm_angle_vel"] = angle_vel
        metrics[f"{side}_forearm_palm_angle_vel_summary"] = _safe_summary(angle_vel)

    return metrics


@torch.no_grad()
def compute_merged_smplx_temporal_metrics(
    smpl_model: torch.nn.Module,
    smpl_params: dict[str, Any],
    fps: float = 30.0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    joints = build_smplx_joints(smpl_model=smpl_model, smpl_params=smpl_params, device=device)
    metrics = compute_temporal_metrics_from_joints(joints, fps=fps)
    metrics.update(compute_forearm_palm_consistency(joints, fps=fps))
    metrics["num_frames"] = int(joints.shape[0])
    return metrics


def compute_prediction_foot_sliding(
    verts: torch.Tensor,
    fps: float = 30.0,
    height_thr: float = 0.03,
    vertical_thr: float = 0.02,
) -> dict[str, Any]:
    if verts.shape[0] < 2:
        empty = np.zeros((0,), dtype=np.float32)
        return {
            "foot_sliding": empty,
            "foot_sliding_summary": _safe_summary(empty),
            "foot_contact_ratio": 0.0,
            "left_foot_sliding": empty,
            "left_foot_sliding_summary": _safe_summary(empty),
            "left_foot_contact_ratio": 0.0,
            "right_foot_sliding": empty,
            "right_foot_sliding_summary": _safe_summary(empty),
            "right_foot_contact_ratio": 0.0,
        }

    metrics: dict[str, Any] = {}
    all_errors = []
    all_contacts = []
    for side, idxs in FOOT_VERTEX_IDXS.items():
        foot_verts = verts[:, idxs, :]
        foot_mean = foot_verts.mean(dim=1)
        ground_height = float(foot_verts[..., 1].min().item())

        heights0 = foot_mean[:-1, 1]
        heights1 = foot_mean[1:, 1]
        delta = foot_mean[1:] - foot_mean[:-1]
        horiz_disp = torch.norm(delta[:, [0, 2]], dim=-1)
        vertical_disp = torch.abs(delta[:, 1])

        contact = (
            (heights0 <= ground_height + height_thr)
            & (heights1 <= ground_height + height_thr)
            & (vertical_disp <= vertical_thr)
        )
        sliding = horiz_disp[contact]

        sliding_np = sliding.cpu().numpy() * 1000.0
        contact_ratio = float(contact.float().mean().item())

        metrics[f"{side}_foot_sliding"] = sliding_np
        metrics[f"{side}_foot_sliding_summary"] = _safe_summary(sliding_np)
        metrics[f"{side}_foot_contact_ratio"] = contact_ratio
        all_contacts.append(contact)
        all_errors.append(sliding_np)

    merged_errors = np.concatenate([x for x in all_errors if x.size > 0]) if any(x.size > 0 for x in all_errors) else np.zeros((0,), dtype=np.float32)
    merged_contact_ratio = float(torch.cat(all_contacts).float().mean().item()) if all_contacts else 0.0
    metrics["foot_sliding"] = merged_errors
    metrics["foot_sliding_summary"] = _safe_summary(merged_errors)
    metrics["foot_contact_ratio"] = merged_contact_ratio
    metrics["foot_sliding_fps"] = float(fps)
    metrics["foot_sliding_height_thr"] = float(height_thr)
    metrics["foot_sliding_vertical_thr"] = float(vertical_thr)
    return metrics


@torch.no_grad()
def compute_merged_smplx_motion_metrics(
    smpl_model: torch.nn.Module,
    smpl_params: dict[str, Any],
    fps: float = 30.0,
    device: str | torch.device = "cpu",
    foot_sliding_height_thr: float = 0.03,
    foot_sliding_vertical_thr: float = 0.02,
) -> dict[str, Any]:
    joints, verts = build_smplx_joints_and_verts(smpl_model=smpl_model, smpl_params=smpl_params, device=device)
    metrics = compute_temporal_metrics_from_joints(joints, fps=fps)
    metrics.update(compute_forearm_palm_consistency(joints, fps=fps))
    metrics.update(
        compute_prediction_foot_sliding(
            verts=verts,
            fps=fps,
            height_thr=foot_sliding_height_thr,
            vertical_thr=foot_sliding_vertical_thr,
        )
    )
    metrics["num_frames"] = int(joints.shape[0])
    return metrics
