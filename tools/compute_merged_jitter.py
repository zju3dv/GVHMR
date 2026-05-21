from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from hmr4d.utils.eval.metric import (
    compute_merged_smplx_motion_metrics,
    compute_merged_smplx_temporal_metrics,
)
from hmr4d.utils.eval.result_loader import load_result, select_smpl_params
from hmr4d.utils.smplx_utils import make_smplx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute temporal jitter/smoothness metrics from merged GVHMR+HaMeR SMPL-X results."
    )
    parser.add_argument("--input", type=str, default=None, help="Single smplx_merged_hamer.pt path.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing per-clip folders with smplx_merged_hamer.pt files.",
    )
    parser.add_argument("--space", choices=("incam", "global"), default="incam")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--include-foot-sliding",
        action="store_true",
        help="Also compute experimental foot sliding metrics. Off by default.",
    )
    parser.add_argument("--foot-height-thr", type=float, default=0.03, help="Ground-contact height threshold in meters.")
    parser.add_argument(
        "--foot-vertical-thr",
        type=float,
        default=0.02,
        help="Per-frame vertical displacement threshold in meters for foot contact.",
    )
    parser.add_argument(
        "--foot-min-contact-ratio",
        type=float,
        default=0.05,
        help="Minimum contact ratio required to treat foot sliding as reliable.",
    )
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV summary output path.")
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if bool(args.input) == bool(args.input_dir):
        raise ValueError("Provide exactly one of --input or --input-dir.")

    if args.input is not None:
        path = Path(args.input)
        if not path.exists():
            raise FileNotFoundError(path)
        return [path]

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    return sorted(input_dir.glob("*/smplx_merged_hamer.pt"))


def flatten_summary(
    result_path: Path,
    loaded: dict,
    metrics: dict,
    space: str,
    foot_min_contact_ratio: float,
    include_foot_sliding: bool,
) -> dict[str, object]:
    row = {
        "clip_id": result_path.parent.name,
        "result_path": str(result_path.resolve()),
        "space": space,
        "result_type": loaded["result_type"],
        "num_frames": metrics["num_frames"],
    }
    merge_meta = loaded.get("cap_merge_meta") or {}
    for key in ("left_hand_frames", "right_hand_frames"):
        if key in merge_meta:
            row[key] = merge_meta[key]
    for key in (
        "gvhmr_preprocess_sec",
        "gvhmr_predict_sec",
        "gvhmr_render_sec",
        "frame_extract_sec",
        "hamer_sec",
        "result_render_sec",
        "pipeline_sec_before_render",
        "pipeline_total_sec",
        "gvhmr_preprocess_fps",
        "gvhmr_predict_fps",
        "frame_extract_fps",
        "hamer_fps",
        "result_render_fps",
        "pipeline_total_fps",
        "hamer_saved_predictions",
    ):
        if key in merge_meta:
            row[key] = merge_meta[key]
    if "left_hand_frames" in row:
        row["left_hand_coverage"] = float(row["left_hand_frames"]) / max(float(row["num_frames"]), 1.0)
    if "right_hand_frames" in row:
        row["right_hand_coverage"] = float(row["right_hand_frames"]) / max(float(row["num_frames"]), 1.0)
    if "left_hand_frames" in row and "right_hand_frames" in row:
        both = min(float(row["left_hand_frames"]), float(row["right_hand_frames"]))
        either = max(float(row["left_hand_frames"]), float(row["right_hand_frames"]))
        row["both_hands_coverage_lower_bound"] = both / max(float(row["num_frames"]), 1.0)
        row["either_hand_coverage_upper_bound"] = either / max(float(row["num_frames"]), 1.0)

    for group in ("fullbody", "body", "left_hand", "right_hand"):
        for metric_name in ("jitter", "accel"):
            summary = metrics[f"{group}_{metric_name}_summary"]
            for stat_name, value in summary.items():
                row[f"{group}_{metric_name}_{stat_name}"] = value
    for side in ("left", "right"):
        for metric_name in ("forearm_palm_angle", "forearm_palm_angle_vel"):
            summary = metrics[f"{side}_{metric_name}_summary"]
            for stat_name, value in summary.items():
                row[f"{side}_{metric_name}_{stat_name}"] = value
    if include_foot_sliding:
        row["foot_metric_mode"] = "experimental"
        row["foot_min_contact_ratio"] = foot_min_contact_ratio
        row["foot_contact_ratio"] = metrics["foot_contact_ratio"]
        row["left_foot_contact_ratio"] = metrics["left_foot_contact_ratio"]
        row["right_foot_contact_ratio"] = metrics["right_foot_contact_ratio"]
        row["foot_sliding_reliable"] = metrics["foot_contact_ratio"] >= foot_min_contact_ratio
        row["left_foot_sliding_reliable"] = metrics["left_foot_contact_ratio"] >= foot_min_contact_ratio
        row["right_foot_sliding_reliable"] = metrics["right_foot_contact_ratio"] >= foot_min_contact_ratio
        for group in ("foot", "left_foot", "right_foot"):
            summary = metrics[f"{group}_sliding_summary"]
            for stat_name, value in summary.items():
                row[f"{group}_sliding_{stat_name}"] = value
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
    result_paths = resolve_inputs(args)
    if not result_paths:
        raise FileNotFoundError("No smplx_merged_hamer.pt files found.")

    smpl_model = make_smplx("supermotion_fullhands").to(args.device).eval()

    rows: list[dict[str, object]] = []
    for result_path in result_paths:
        loaded = load_result(result_path)
        smpl_params = select_smpl_params(loaded, space=args.space)
        if args.include_foot_sliding:
            metrics = compute_merged_smplx_motion_metrics(
                smpl_model=smpl_model,
                smpl_params=smpl_params,
                fps=args.fps,
                device=args.device,
                foot_sliding_height_thr=args.foot_height_thr,
                foot_sliding_vertical_thr=args.foot_vertical_thr,
            )
        else:
            metrics = compute_merged_smplx_temporal_metrics(
                smpl_model=smpl_model,
                smpl_params=smpl_params,
                fps=args.fps,
                device=args.device,
            )
        row = flatten_summary(
            result_path,
            loaded,
            metrics,
            args.space,
            foot_min_contact_ratio=args.foot_min_contact_ratio,
            include_foot_sliding=args.include_foot_sliding,
        )
        rows.append(row)
        if args.include_foot_sliding:
            print(
                f"{row['clip_id']}: "
                f"fullbody_jitter={row['fullbody_jitter_mean']:.4f}, "
                f"left_hand_jitter={row['left_hand_jitter_mean']:.4f}, "
                f"right_hand_jitter={row['right_hand_jitter_mean']:.4f}, "
                f"left_wrist_boundary={row['left_forearm_palm_angle_vel_mean']:.4f}, "
                f"right_wrist_boundary={row['right_forearm_palm_angle_vel_mean']:.4f}, "
                f"foot_contact_ratio={row['foot_contact_ratio']:.4f}, "
                f"foot_sliding={row['foot_sliding_mean']:.4f}, "
                f"foot_sliding_reliable={row['foot_sliding_reliable']}"
            )
        else:
            print(
                f"{row['clip_id']}: "
                f"left_cov={row.get('left_hand_coverage', 0.0):.3f}, "
                f"right_cov={row.get('right_hand_coverage', 0.0):.3f}, "
                f"fullbody_jitter={row['fullbody_jitter_mean']:.4f}, "
                f"left_hand_jitter={row['left_hand_jitter_mean']:.4f}, "
                f"right_hand_jitter={row['right_hand_jitter_mean']:.4f}, "
                f"left_wrist_boundary={row['left_forearm_palm_angle_vel_mean']:.4f}, "
                f"right_wrist_boundary={row['right_forearm_palm_angle_vel_mean']:.4f}, "
                f"pipeline_fps={row.get('pipeline_total_fps', 0.0):.3f}"
            )

    if args.output_csv is not None:
        write_csv(rows, Path(args.output_csv))
        print(f"Saved CSV: {Path(args.output_csv).resolve()}")


if __name__ == "__main__":
    main()
