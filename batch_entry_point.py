import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from hmr4d.utils.pylogger import Log

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HAMER_ROOT = REPO_ROOT / "hamer"


def parse_accept(value: str) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "accept", "accepted"}


def load_accept_map(csv_path: Path) -> dict[str, bool]:
    accept_map: dict[str, bool] = {}
    if not csv_path.exists():
        raise FileNotFoundError(f"quality label csv not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "clip_id" not in reader.fieldnames or "accept" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain columns: clip_id, accept")
        for row in reader:
            clip_id = str(row["clip_id"]).strip()
            accept_map[clip_id] = parse_accept(row["accept"])
    return accept_map


def build_entry_point_cmd(args: argparse.Namespace, clip_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "entry_point.py"),
        "--video",
        str(clip_path),
        "--output-root",
        str(args.output_root),
        "--person-select-ui",
        args.person_select_ui,
    ]
    if args.static_cam:
        cmd.append("--static-cam")
    if args.use_dpvo:
        cmd.append("--use-dpvo")
    if args.f_mm is not None:
        cmd.extend(["--f-mm", str(args.f_mm)])
    if args.auto_person:
        cmd.append("--auto-person")
    if args.person_track_id is not None:
        cmd.extend(["--person-track-id", str(args.person_track_id)])
    if args.verbose:
        cmd.append("--verbose")
    if args.render_preview:
        cmd.append("--render-preview")
    if args.force:
        cmd.append("--force")
    if args.skip_result_video:
        cmd.append("--skip-result-video")
    if args.no_interactive:
        cmd.append("--no-interactive")
    if args.hamer_root is not None:
        cmd.extend(["--hamer-root", str(args.hamer_root)])
    if args.hamer_checkpoint is not None:
        cmd.extend(["--hamer-checkpoint", str(args.hamer_checkpoint)])
    if args.hamer_batch_size is not None:
        cmd.extend(["--hamer-batch-size", str(args.hamer_batch_size)])
    if args.hamer_rescale_factor is not None:
        cmd.extend(["--hamer-rescale-factor", str(args.hamer_rescale_factor)])
    if args.hand_min_conf is not None:
        cmd.extend(["--hand-min-conf", str(args.hand_min_conf)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run entry_point.py for accepted tennis clips in batch."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="inputs/tennis_custom",
        help="Root path that contains clips/ and meta/quality_labels.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/cap_pipeline/tennis_custom",
        help="Root output directory passed to entry_point.py.",
    )
    parser.add_argument(
        "--quality-csv",
        type=str,
        default=None,
        help="Optional path to quality_labels.csv. Defaults to <dataset-root>/meta/quality_labels.csv.",
    )
    parser.add_argument(
        "--only-accept",
        action="store_true",
        help="Run only clips with accept=True. Enabled by default unless --all-clips is set.",
    )
    parser.add_argument(
        "--all-clips",
        action="store_true",
        help="Ignore quality_labels.csv accept values and run every clip.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip clip if <output-root>/<clip_id>/smplx_merged_hamer.pt already exists.",
    )
    parser.add_argument("--static-cam", action="store_true")
    parser.add_argument("--use-dpvo", action="store_true")
    parser.add_argument("--f-mm", type=int, default=None)
    parser.add_argument("--auto-person", action="store_true")
    parser.add_argument("--person-select-ui", choices=("auto", "window", "terminal"), default="auto")
    parser.add_argument("--person-track-id", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--render-preview", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-result-video", action="store_true")
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--hamer-root", type=str, default=str(DEFAULT_HAMER_ROOT))
    parser.add_argument("--hamer-checkpoint", type=str, default=None)
    parser.add_argument("--hamer-batch-size", type=int, default=1)
    parser.add_argument("--hamer-rescale-factor", type=float, default=2.5)
    parser.add_argument("--hand-min-conf", type=float, default=0.35)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    clips_dir = dataset_root / "clips"
    quality_csv = Path(args.quality_csv) if args.quality_csv else dataset_root / "meta" / "quality_labels.csv"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not clips_dir.exists():
        raise FileNotFoundError(f"clips dir not found: {clips_dir}")

    filter_accept = not args.all_clips
    accept_map = load_accept_map(quality_csv) if filter_accept else {}
    if filter_accept:
        Log.info(f"Loaded {len(accept_map)} rows from {quality_csv}")

    clip_paths = sorted(list(clips_dir.glob("*.mp4")) + list(clips_dir.glob("*.MP4")))
    Log.info(f"Found {len(clip_paths)} clips in {clips_dir}")

    total = 0
    skipped = 0
    failed = 0
    env = dict(os.environ)
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if old_pythonpath == "" else f"{REPO_ROOT}:{old_pythonpath}"
    for clip_path in clip_paths:
        clip_id = clip_path.stem
        total += 1

        if filter_accept and not accept_map.get(clip_id, False):
            skipped += 1
            Log.info(f"[Skip] {clip_id} not accepted in {quality_csv.name}")
            continue

        if args.skip_existing:
            merged_path = output_root / clip_id / "smplx_merged_hamer.pt"
            if merged_path.exists():
                skipped += 1
                Log.info(f"[Skip] {clip_id} already has merged result: {merged_path}")
                continue

        cmd = build_entry_point_cmd(args, clip_path)
        Log.info(f"[Run] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
        except subprocess.CalledProcessError:
            failed += 1
            Log.error(f"[Fail] clip_id={clip_id}, path={clip_path}")

    Log.info(
        f"Done. total={total}, skipped={skipped}, failed={failed}, success={total - skipped - failed}"
    )


if __name__ == "__main__":
    main()
