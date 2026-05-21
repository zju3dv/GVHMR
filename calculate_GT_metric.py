from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
CAP_ROOT = REPO_ROOT / "CAP"
if str(CAP_ROOT) not in sys.path:
    sys.path.insert(0, str(CAP_ROOT))

from hmr4d.utils.eval.eval_utils import batch_compute_similarity_transform_torch, compute_jitter
from hmr4d.utils.eval.metric import (
    BODY_JOINT_IDXS,
    FULL_BODY_JOINT_IDXS,
    LEFT_HAND_JOINT_IDXS,
    RIGHT_HAND_JOINT_IDXS,
    build_smplx_joints,
)
from hmr4d.utils.eval.result_loader import load_result, torch_load_file
from hmr4d.utils.smplx_utils import make_smplx


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

JOINT_GROUPS = {
    "fullbody": FULL_BODY_JOINT_IDXS,
    "body": BODY_JOINT_IDXS,
    "left_hand": LEFT_HAND_JOINT_IDXS,
    "right_hand": RIGHT_HAND_JOINT_IDXS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute MPJPE and jitter for target SMPL-X parameters against GT SMPL-X npz/pt animations."
        )
    )
    parser.add_argument("--gt", type=str, default=None, help="Single GT .npz/.pt parameter file.")
    parser.add_argument("--target", type=str, default=None, help="Single target .npz/.pt result or parameter file.")
    parser.add_argument("--gt-dir", type=str, default=None, help="Directory containing GT .npz/.pt files.")
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Directory containing target files or per-clip folders with smplx_merged_hamer.pt.",
    )
    parser.add_argument("--target-name", type=str, default="smplx_merged_hamer.pt")
    parser.add_argument("--space", choices=("incam", "global"), default="global")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--align-pelvis", action="store_true", help="Pelvis-align joints before MPJPE.")
    parser.add_argument("--pa-mpjpe", action="store_true", help="Also compute Procrustes-aligned MPJPE.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def to_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def infer_num_frames(params: dict[str, torch.Tensor]) -> int:
    for key in ("global_orient", "body_pose", "transl", "left_hand_pose", "right_hand_pose"):
        value = params.get(key)
        if value is not None and value.ndim >= 2:
            return int(value.shape[0])
    raise ValueError("Could not infer frame count from SMPL-X parameters.")


def normalize_smplx_params(raw_params: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    params: dict[str, torch.Tensor] = {}
    for key in SMPLX_PARAM_KEYS:
        if key in raw_params and raw_params[key] is not None:
            params[key] = to_tensor(raw_params[key], device)

    num_frames = infer_num_frames(params)
    defaults = {
        "global_orient": (num_frames, 3),
        "body_pose": (num_frames, 63),
        "transl": (num_frames, 3),
        "left_hand_pose": (num_frames, 45),
        "right_hand_pose": (num_frames, 45),
        "jaw_pose": (num_frames, 3),
        "leye_pose": (num_frames, 3),
        "reye_pose": (num_frames, 3),
        "expression": (num_frames, 10),
    }
    for key, shape in defaults.items():
        if key not in params:
            params[key] = torch.zeros(shape, dtype=torch.float32, device=device)

    betas = params.get("betas")
    if betas is None:
        params["betas"] = torch.zeros((num_frames, 10), dtype=torch.float32, device=device)
    elif betas.ndim == 1:
        params["betas"] = betas[None].expand(num_frames, -1).contiguous()
    elif betas.shape[0] != num_frames:
        params["betas"] = betas[:1].expand(num_frames, -1).contiguous()

    for key, value in list(params.items()):
        if value.ndim == 1 and key != "betas":
            params[key] = value[None].expand(num_frames, -1).contiguous()
    return params


def unwrap_npz_value(value: np.ndarray) -> Any:
    if value.dtype == object and value.shape == ():
        return value.item()
    return value


def load_npz_params(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    data = np.load(path, allow_pickle=True)
    raw = {key: unwrap_npz_value(data[key]) for key in data.files if key in SMPLX_PARAM_KEYS}
    return normalize_smplx_params(raw, device)


def pick_param_dict(data: dict[str, Any], space: str) -> dict[str, Any]:
    if f"smpl_params_{space}" in data:
        return data[f"smpl_params_{space}"]
    if "smpl_params" in data:
        return data["smpl_params"]
    if all(key in data for key in ("global_orient", "body_pose", "transl")):
        return data
    raw = data.get("raw")
    if isinstance(raw, dict):
        return pick_param_dict(raw, space)
    raise KeyError(f"Could not find SMPL-X parameters in keys: {sorted(data.keys())}")


def load_any_params(path: Path, device: torch.device, space: str) -> dict[str, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_npz_params(path, device)
    if suffix in {".pt", ".pth"}:
        try:
            loaded = load_result(path)
            raw_params = loaded.get(f"smpl_params_{space}") or loaded.get("smpl_params_global") or loaded.get("smpl_params_incam")
        except (KeyError, TypeError):
            raw = torch_load_file(path, map_location="cpu")
            if not isinstance(raw, dict):
                raise TypeError(f"Expected dict in {path}, got {type(raw)}")
            raw_params = pick_param_dict(raw, space)
        if raw_params is None:
            raise KeyError(f"Missing target SMPL-X params for space={space}: {path}")
        return normalize_smplx_params(raw_params, device)
    raise ValueError(f"Unsupported parameter file type: {path}")


def strip_stageii_suffix(name: str) -> str:
    return name[:-8] if name.endswith("_stageii") else name


def clip_id_from_gt(path: Path) -> str:
    return strip_stageii_suffix(path.stem)


def clip_id_from_target(path: Path, target_name: str) -> str:
    name = path.parent.name if path.name == target_name else path.stem
    return strip_stageii_suffix(name)


def collect_gt_paths(gt_dir: Path) -> dict[str, Path]:
    paths = sorted([*gt_dir.glob("*.npz"), *gt_dir.glob("*.pt"), *gt_dir.glob("*.pth")])
    return {clip_id_from_gt(path): path for path in paths}


def collect_target_paths(target_dir: Path, target_name: str) -> dict[str, Path]:
    nested = sorted(target_dir.glob(f"*/{target_name}"))
    direct = sorted([*target_dir.glob("*.npz"), *target_dir.glob("*.pt"), *target_dir.glob("*.pth")])
    paths = nested if nested else direct
    return {clip_id_from_target(path, target_name): path for path in paths}


def resolve_pairs(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    single_mode = args.gt is not None or args.target is not None
    dir_mode = args.gt_dir is not None or args.target_dir is not None
    if single_mode == dir_mode:
        raise ValueError("Use exactly one mode: (--gt and --target) or (--gt-dir and --target-dir).")

    if single_mode:
        if args.gt is None or args.target is None:
            raise ValueError("Single-file mode requires both --gt and --target.")
        gt_path = Path(args.gt)
        target_path = Path(args.target)
        if not gt_path.exists():
            raise FileNotFoundError(gt_path)
        if not target_path.exists():
            raise FileNotFoundError(target_path)
        return [(clip_id_from_gt(gt_path), gt_path, target_path)]

    gt_dir = Path(args.gt_dir)
    target_dir = Path(args.target_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(gt_dir)
    if not target_dir.exists():
        raise FileNotFoundError(target_dir)

    gt_paths = collect_gt_paths(gt_dir)
    target_paths = collect_target_paths(target_dir, args.target_name)
    common_ids = sorted(set(gt_paths) & set(target_paths))
    if not common_ids:
        raise FileNotFoundError(
            f"No matching clip ids. GT ids={len(gt_paths)}, target ids={len(target_paths)}"
        )
    return [(clip_id, gt_paths[clip_id], target_paths[clip_id]) for clip_id in common_ids]


def safe_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def pelvis_align(joints: torch.Tensor, pelvis_idxs: tuple[int, int] = (1, 2)) -> torch.Tensor:
    pelvis = joints[:, pelvis_idxs, :].mean(dim=1, keepdim=True)
    return joints - pelvis


def compute_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> np.ndarray:
    return torch.linalg.norm(pred - gt, dim=-1).mean(dim=-1).cpu().numpy() * 1000.0


def compute_group_metrics(
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    fps: float,
    align_pelvis_for_mpjpe: bool,
    include_pa_mpjpe: bool,
) -> dict[str, Any]:
    num_frames = min(pred_joints.shape[0], gt_joints.shape[0])
    pred_joints = pred_joints[:num_frames]
    gt_joints = gt_joints[:num_frames]

    mpjpe_pred_joints = pelvis_align(pred_joints) if align_pelvis_for_mpjpe else pred_joints
    mpjpe_gt_joints = pelvis_align(gt_joints) if align_pelvis_for_mpjpe else gt_joints

    metrics: dict[str, Any] = {"num_frames": num_frames}
    for group_name, idxs in JOINT_GROUPS.items():
        pred_group = pred_joints[:, idxs, :]
        gt_group = gt_joints[:, idxs, :]
        mpjpe = compute_mpjpe(mpjpe_pred_joints[:, idxs, :], mpjpe_gt_joints[:, idxs, :])
        pred_jitter = compute_jitter(pred_group, fps=fps)
        gt_jitter = compute_jitter(gt_group, fps=fps)
        jitter_delta = pred_jitter - gt_jitter
        jitter_abs_delta = np.abs(jitter_delta)

        metrics[f"{group_name}_mpjpe"] = mpjpe
        metrics[f"{group_name}_mpjpe_summary"] = safe_summary(mpjpe)
        metrics[f"{group_name}_target_jitter"] = pred_jitter
        metrics[f"{group_name}_target_jitter_summary"] = safe_summary(pred_jitter)
        metrics[f"{group_name}_gt_jitter"] = gt_jitter
        metrics[f"{group_name}_gt_jitter_summary"] = safe_summary(gt_jitter)
        metrics[f"{group_name}_jitter_delta"] = jitter_delta
        metrics[f"{group_name}_jitter_delta_summary"] = safe_summary(jitter_delta)
        metrics[f"{group_name}_jitter_abs_delta"] = jitter_abs_delta
        metrics[f"{group_name}_jitter_abs_delta_summary"] = safe_summary(jitter_abs_delta)

        if include_pa_mpjpe:
            pa_pred = batch_compute_similarity_transform_torch(pred_group.cpu(), gt_group.cpu())
            pa_mpjpe = compute_mpjpe(pa_pred, gt_group.cpu())
            metrics[f"{group_name}_pa_mpjpe"] = pa_mpjpe
            metrics[f"{group_name}_pa_mpjpe_summary"] = safe_summary(pa_mpjpe)
    return metrics


def flatten_row(
    clip_id: str,
    gt_path: Path,
    target_path: Path,
    metrics: dict[str, Any],
    space: str,
    align_pelvis_for_mpjpe: bool,
    include_pa_mpjpe: bool,
) -> dict[str, object]:
    row: dict[str, object] = {
        "clip_id": clip_id,
        "gt_path": str(gt_path.resolve()),
        "target_path": str(target_path.resolve()),
        "space": space,
        "num_frames": metrics["num_frames"],
        "mpjpe_aligned_by_pelvis": align_pelvis_for_mpjpe,
    }
    metric_names = ["mpjpe", "target_jitter", "gt_jitter", "jitter_delta", "jitter_abs_delta"]
    if include_pa_mpjpe:
        metric_names.append("pa_mpjpe")

    for group_name in JOINT_GROUPS:
        for metric_name in metric_names:
            summary = metrics[f"{group_name}_{metric_name}_summary"]
            for stat_name, value in summary.items():
                row[f"{group_name}_{metric_name}_{stat_name}"] = value
    return row


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pairs = resolve_pairs(args)
    device = torch.device(args.device)
    smpl_model = make_smplx("supermotion_fullhands").to(device).eval()

    rows: list[dict[str, object]] = []
    for clip_id, gt_path, target_path in pairs:
        gt_params = load_any_params(gt_path, device=device, space=args.space)
        target_params = load_any_params(target_path, device=device, space=args.space)
        gt_joints = build_smplx_joints(smpl_model, gt_params, device=device)
        target_joints = build_smplx_joints(smpl_model, target_params, device=device)
        metrics = compute_group_metrics(
            pred_joints=target_joints,
            gt_joints=gt_joints,
            fps=args.fps,
            align_pelvis_for_mpjpe=args.align_pelvis,
            include_pa_mpjpe=args.pa_mpjpe,
        )
        row = flatten_row(
            clip_id=clip_id,
            gt_path=gt_path,
            target_path=target_path,
            metrics=metrics,
            space=args.space,
            align_pelvis_for_mpjpe=args.align_pelvis,
            include_pa_mpjpe=args.pa_mpjpe,
        )
        rows.append(row)
        print(
            f"{clip_id}: "
            f"frames={row['num_frames']}, "
            f"body_mpjpe={row['body_mpjpe_mean']:.2f}mm, "
            f"fullbody_mpjpe={row['fullbody_mpjpe_mean']:.2f}mm, "
            f"target_jitter={row['fullbody_target_jitter_mean']:.4f}, "
            f"gt_jitter={row['fullbody_gt_jitter_mean']:.4f}"
        )

    if args.output_csv is not None:
        write_csv(rows, Path(args.output_csv))
        print(f"Saved CSV: {Path(args.output_csv).resolve()}")


if __name__ == "__main__":
    main()
