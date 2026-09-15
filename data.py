#!/usr/bin/env python3
"""
Utilities for tennis custom dataset.

1) Build dataset clips/metadata from raw videos.
2) Run single-clip demo command across all clips (batch mode).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
PREPROCESS_FILES = ["bbx.pt", "vitpose.pt", "vit_features.pt", "slam_results.pt"]


@dataclass
class ClipMeta:
    clip_id: str
    source_video: str
    start_frame: int
    end_frame: int
    fps: float
    width: int
    height: int


def run_cmd(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def ffprobe_video(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    out = run_cmd(cmd)
    data = json.loads(out)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found: {video_path}")
    return streams[0]


def parse_rate(rate: str) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        a, b = rate.split("/")
        a_f = float(a)
        b_f = float(b)
        return a_f / b_f if b_f != 0 else 0.0
    return float(rate)


def copy_sources_to_raw(source_paths: List[Path], raw_dir: Path) -> List[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in source_paths:
        dst = raw_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def create_clip(
    src_video: Path,
    out_clip: Path,
    start_sec: float,
    dur_sec: float,
    fps: float,
) -> None:
    out_clip.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(src_video),
        "-t",
        f"{dur_sec:.6f}",
        "-an",
        "-r",
        f"{fps:.6f}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_clip),
    ]
    run_cmd(cmd)


def touch_preprocess_files(clip_output_dir: Path) -> None:
    preprocess = clip_output_dir / "preprocess"
    preprocess.mkdir(parents=True, exist_ok=True)
    for name in PREPROCESS_FILES:
        p = preprocess / name
        if not p.exists():
            p.touch()


def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def collect_videos(source_globs: List[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in source_globs:
        for p in sorted(glob.glob(pattern)):
            path = Path(p)
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                files.append(path)
    unique = []
    seen = set()
    for p in files:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def build_dataset(args: argparse.Namespace) -> None:
    if args.clip_seconds <= 0:
        raise ValueError("--clip-seconds must be > 0")

    in_root = Path(args.inputs_root)
    raw_dir = in_root / "raw_videos"
    clips_dir = in_root / "clips"
    meta_dir = in_root / "meta"
    out_root = Path(args.outputs_root)

    clips_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    source_videos = collect_videos(args.source_glob)
    if not source_videos:
        raise FileNotFoundError(
            "No source videos found. Use --source-glob to point to your raw videos."
        )

    if args.skip_copy_to_raw:
        raw_videos = source_videos
    else:
        raw_videos = copy_sources_to_raw(source_videos, raw_dir)

    all_meta: List[ClipMeta] = []
    quality_rows: List[List[object]] = []

    clip_num = args.start_clip_id
    for video in raw_videos:
        stream = ffprobe_video(video)
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))

        fps = parse_rate(stream.get("avg_frame_rate", "0/0"))
        if fps <= 0:
            fps = parse_rate(stream.get("r_frame_rate", "0/0"))
        if fps <= 0:
            fps = 30.0

        nb_frames = stream.get("nb_frames")
        duration = float(stream.get("duration", 0) or 0)
        if duration <= 0 and nb_frames and fps > 0:
            duration = int(nb_frames) / fps
        if duration <= 0:
            continue

        n_segments = int(duration // args.clip_seconds)
        if duration % args.clip_seconds > 1e-6:
            n_segments += 1

        for i in range(n_segments):
            start_sec = i * args.clip_seconds
            remain = max(0.0, duration - start_sec)
            seg_dur = min(args.clip_seconds, remain)
            if seg_dur <= 0.05:
                continue

            clip_id = f"{clip_num:06d}"
            clip_path = clips_dir / f"{clip_id}.mp4"
            create_clip(video, clip_path, start_sec, seg_dur, fps)

            start_frame = int(round(start_sec * fps))
            end_frame = max(start_frame, int(round((start_sec + seg_dur) * fps)) - 1)

            all_meta.append(
                ClipMeta(
                    clip_id=clip_id,
                    source_video=video.name,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    fps=round(fps, 6),
                    width=width,
                    height=height,
                )
            )
            quality_rows.append([clip_id, 1, ""])

            # Keep requested output structure as placeholder.
            clip_out = out_root / clip_id
            clip_out.mkdir(parents=True, exist_ok=True)
            target = clip_out / "0_input_video.mp4"
            if not target.exists():
                shutil.copy2(clip_path, target)
            touch_preprocess_files(clip_out)

            clip_num += 1

    clip_rows = [
        [
            c.clip_id,
            c.source_video,
            c.start_frame,
            c.end_frame,
            f"{c.fps:.6f}",
            c.width,
            c.height,
        ]
        for c in all_meta
    ]

    write_csv(
        meta_dir / "clips.csv",
        [
            "clip_id",
            "source_video",
            "start_frame",
            "end_frame",
            "fps",
            "width",
            "height",
        ],
        clip_rows,
    )

    write_csv(
        meta_dir / "quality_labels.csv",
        ["clip_id", "accept", "reason"],
        quality_rows,
    )

    write_csv(
        meta_dir / "racket_labels.csv",
        [
            "clip_id",
            "frame_idx",
            "racket_bbox",
            "tip",
            "handle",
            "stroke_type",
            "contact",
        ],
        [],
    )

    print(f"Done. Source videos: {len(raw_videos)}, clips: {len(all_meta)}")
    print(f"inputs: {in_root}")
    print(f"outputs: {out_root}")


def run_demo_batch(args: argparse.Namespace) -> None:
    clips_dir = Path(args.clips_dir)
    outputs_root = Path(args.outputs_root)

    clip_paths = sorted(clips_dir.glob("*.mp4"))
    if not clip_paths:
        raise FileNotFoundError(f"No clips found in: {clips_dir}")

    if not args.command_template and not args.prepare_only:
        raise ValueError(
            "Provide --command-template for real demo execution, or use --prepare-only."
        )

    outputs_root.mkdir(parents=True, exist_ok=True)

    total = len(clip_paths)
    processed = 0
    skipped = 0
    failed = 0

    for idx, clip_path in enumerate(clip_paths, start=1):
        clip_id = clip_path.stem
        clip_out_dir = outputs_root / clip_id
        clip_out_dir.mkdir(parents=True, exist_ok=True)

        out_video = clip_out_dir / "0_input_video.mp4"
        if args.skip_existing and out_video.exists():
            skipped += 1
            print(f"[{idx}/{total}] skip {clip_id} (already exists)")
            continue

        if out_video.exists() or out_video.is_symlink():
            out_video.unlink()

        if args.link_input:
            out_video.symlink_to(clip_path.resolve())
        else:
            shutil.copy2(clip_path, out_video)

        touch_preprocess_files(clip_out_dir)

        if args.prepare_only:
            processed += 1
            print(f"[{idx}/{total}] prepared {clip_id}")
            continue

        input_q = shlex.quote(str(clip_path.resolve()))
        output_q = shlex.quote(str(clip_out_dir.resolve()))
        clip_id_q = shlex.quote(clip_id)

        cmd = (
            args.command_template.replace("{input}", input_q)
            .replace("{output}", output_q)
            .replace("{clip_id}", clip_id_q)
        )

        proc = subprocess.run(cmd, shell=True)
        if proc.returncode != 0:
            failed += 1
            print(f"[{idx}/{total}] failed {clip_id} (exit={proc.returncode})")
            if args.stop_on_error:
                break
            continue

        processed += 1
        print(f"[{idx}/{total}] done {clip_id}")

    print(
        "Batch finished. "
        f"processed={processed}, skipped={skipped}, failed={failed}, total={total}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build tennis dataset and run demo across all clips."
    )
    subparsers = parser.add_subparsers(dest="mode")

    p_build = subparsers.add_parser("build", help="Create dataset clips and metadata.")
    p_build.add_argument(
        "--clip-seconds",
        type=float,
        default=5.0,
        help="Target clip length in seconds (default: 5.0)",
    )
    p_build.add_argument(
        "--source-glob",
        action="append",
        default=["*.mp4", "*.mov", "*.mkv", "*.avi", "*.m4v"],
        help="Glob pattern(s) for source videos. Can be repeated.",
    )
    p_build.add_argument(
        "--skip-copy-to-raw",
        action="store_true",
        help="Use source videos directly instead of copying into raw_videos.",
    )
    p_build.add_argument(
        "--start-clip-id",
        type=int,
        default=1,
        help="Starting numeric clip id (default: 1)",
    )
    p_build.add_argument(
        "--inputs-root",
        default="inputs/tennis_custom",
        help="Input dataset root (default: inputs/tennis_custom)",
    )
    p_build.add_argument(
        "--outputs-root",
        default="outputs/demo/tennis_custom",
        help="Output root (default: outputs/demo/tennis_custom)",
    )
    p_build.set_defaults(func=build_dataset)

    p_demo = subparsers.add_parser(
        "demo-batch",
        help="Run single-clip demo command for every clip in dataset.",
    )
    p_demo.add_argument(
        "--clips-dir",
        default="inputs/tennis_custom/clips",
        help="Directory containing clip mp4 files.",
    )
    p_demo.add_argument(
        "--outputs-root",
        default="outputs/demo/tennis_custom",
        help="Per-clip output root directory.",
    )
    p_demo.add_argument(
        "--command-template",
        default="",
        help=(
            "Single-clip demo command template. "
            "Use placeholders: {input}, {output}, {clip_id}."
        ),
    )
    p_demo.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip clips that already have 0_input_video.mp4",
    )
    p_demo.add_argument(
        "--link-input",
        action="store_true",
        help="Use symlink for 0_input_video.mp4 instead of copying clip",
    )
    p_demo.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare output folders/files without running demo command",
    )
    p_demo.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop batch at first failed clip",
    )
    p_demo.set_defaults(func=run_demo_batch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Backward compatibility: default to build mode when no subcommand is given.
    if not getattr(args, "mode", None):
        args.mode = "build"
        args.inputs_root = "inputs/tennis_custom"
        args.outputs_root = "outputs/demo/tennis_custom"
        args.func = build_dataset

    args.func(args)


if __name__ == "__main__":
    main()
