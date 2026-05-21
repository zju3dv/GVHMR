import argparse
import csv
import os
import pickle
import subprocess
import sys
from pathlib import Path

import torch
from hmr4d.utils.pylogger import Log


def check_required_checkpoints():
    required = [
        Path("inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"),
        Path("inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"),
        Path("inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"),
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        missing_str = "\n".join([f"- {p}" for p in missing])
        raise FileNotFoundError(
            "Missing required checkpoints:\n"
            f"{missing_str}\n"
            "Please place these files before running batch inference."
        )


def is_corrupted_pt(path: Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size == 0:
        return True
    try:
        torch.load(path)
        return False
    except (EOFError, RuntimeError, ValueError, pickle.UnpicklingError):
        return True


def cleanup_corrupted_cache(output_root: Path, clip_id: str) -> int:
    clip_dir = output_root / clip_id
    candidate_files = [
        clip_dir / "preprocess" / "bbx.pt",
        clip_dir / "preprocess" / "vitpose.pt",
        clip_dir / "preprocess" / "vit_features.pt",
        clip_dir / "preprocess" / "slam_results.pt",
        clip_dir / "hmr4d_results.pt",
    ]
    removed = 0
    for f in candidate_files:
        if is_corrupted_pt(f):
            f.unlink(missing_ok=True)
            removed += 1
            Log.warning(f"[Cleanup] removed corrupted cache: {f}")
    return removed


def cleanup_clip_cache(output_root: Path, clip_id: str):
    clip_dir = output_root / clip_id
    targets = [
        clip_dir / "preprocess" / "bbx.pt",
        clip_dir / "preprocess" / "vitpose.pt",
        clip_dir / "preprocess" / "vit_features.pt",
        clip_dir / "preprocess" / "slam_results.pt",
        clip_dir / "hmr4d_results.pt",
        clip_dir / "1_incam.mp4",
        clip_dir / "2_global.mp4",
        clip_dir / f"{clip_id}_3_incam_global_horiz.mp4",
    ]
    for t in targets:
        t.unlink(missing_ok=True)


def parse_accept(value: str) -> bool:
    v = str(value).strip().lower()
    return v in {"1", "true", "t", "yes", "y", "accept", "accepted"}


def load_accept_map(csv_path: Path) -> dict:
    accept_map = {}
    if not csv_path.exists():
        return accept_map

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "clip_id" not in reader.fieldnames or "accept" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain columns: clip_id, accept")
        for row in reader:
            clip_id = str(row["clip_id"]).strip()
            accept = parse_accept(row["accept"])
            accept_map[clip_id] = accept
    return accept_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="inputs/tennis_custom",
        help="Root path that contains raw_videos/, clips/, meta/",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/demo/tennis_custom",
        help="Output root for demo.py",
    )
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip VO")
    parser.add_argument("--use_dpvo", action="store_true", help="Use DPVO instead of SimpleVO")
    parser.add_argument("--f_mm", type=int, default=None, help="Fullframe focal length in mm")
    parser.add_argument("--verbose", action="store_true", help="Pass verbose flag to demo.py")
    parser.add_argument(
        "--only_accept",
        action="store_true",
        help="Run only clips with accept=True in meta/quality_labels.csv",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip clip if <output_root>/<clip_id>/hmr4d_results.pt already exists",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    clips_dir = dataset_root / "clips"
    meta_dir = dataset_root / "meta"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    check_required_checkpoints()

    if not clips_dir.exists():
        raise FileNotFoundError(f"clips dir not found: {clips_dir}")

    accept_map = {}
    if args.only_accept:
        quality_csv = meta_dir / "quality_labels.csv"
        accept_map = load_accept_map(quality_csv)
        Log.info(f"Loaded {len(accept_map)} rows from {quality_csv}")

    clip_paths = sorted(list(clips_dir.glob("*.mp4")) + list(clips_dir.glob("*.MP4")))
    Log.info(f"Found {len(clip_paths)} clips in {clips_dir}")

    total = 0
    skipped = 0
    failed = 0
    for clip_path in clip_paths:
        clip_id = clip_path.stem
        total += 1

        if args.only_accept and not accept_map.get(clip_id, False):
            skipped += 1
            Log.info(f"[Skip] {clip_id} not accepted in quality_labels.csv")
            continue

        if args.skip_existing:
            result_pt = output_root / clip_id / "hmr4d_results.pt"
            if result_pt.exists():
                skipped += 1
                Log.info(f"[Skip] {clip_id} already has result: {result_pt}")
                continue

        cleanup_corrupted_cache(output_root, clip_id)

        cmd = [
            sys.executable,
            "tools/demo/demo.py",
            "--video",
            str(clip_path),
            "--output_root",
            str(output_root),
        ]
        if args.static_cam:
            cmd.append("-s")
        if args.use_dpvo:
            cmd.append("--use_dpvo")
        if args.f_mm is not None:
            cmd += ["--f_mm", str(args.f_mm)]
        if args.verbose:
            cmd.append("--verbose")

        Log.info(f"[Run] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, env=dict(os.environ))
        except subprocess.CalledProcessError:
            Log.warning(f"[Retry] clip_id={clip_id}: cleaning clip cache and retrying once")
            cleanup_clip_cache(output_root, clip_id)
            try:
                subprocess.run(cmd, check=True, env=dict(os.environ))
            except subprocess.CalledProcessError:
                failed += 1
                Log.error(f"[Fail] clip_id={clip_id}, path={clip_path}")

    Log.info(
        f"Done. total={total}, skipped={skipped}, failed={failed}, success={total - skipped - failed}"
    )


if __name__ == "__main__":
    main()
