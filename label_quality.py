#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_quality_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "accept", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def load_clip_ids(clips_csv: Path) -> List[str]:
    rows = read_csv_rows(clips_csv)
    ids = []
    for row in rows:
        clip_id = row.get("clip_id", "").strip()
        if clip_id:
            ids.append(clip_id)
    return ids


def load_quality_map(quality_csv: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(quality_csv)
    m: Dict[str, Dict[str, str]] = {}
    for row in rows:
        clip_id = row.get("clip_id", "").strip()
        if clip_id:
            m[clip_id] = {
                "clip_id": clip_id,
                "accept": row.get("accept", "").strip(),
                "reason": row.get("reason", "").strip(),
            }
    return m


def build_quality_rows(clip_ids: List[str], quality_map: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    rows = []
    for clip_id in clip_ids:
        if clip_id in quality_map:
            rows.append(quality_map[clip_id])
        else:
            rows.append({"clip_id": clip_id, "accept": "", "reason": ""})
    return rows


def find_start_index(rows: List[Dict[str, str]]) -> int:
    for i, row in enumerate(rows):
        if row["accept"] == "":
            return i
    return 0


def put_text(img, text: str, y: int) -> int:
    cv2.putText(
        img,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return y + 32


def draw_overlay(frame, clip_id: str, idx: int, total: int, status: str, paused: bool):
    img = frame.copy()
    y = 35
    y = put_text(img, f"clip_id: {clip_id} ({idx + 1}/{total})", y)
    y = put_text(img, f"label: {status}", y)
    y = put_text(img, "a/1=accept  d/0=reject  s=skip  b=back  space=pause  q=quit", y)
    if paused:
        y = put_text(img, "[PAUSED]", y)
    return img


def open_video_info(video_path: Path) -> Tuple[cv2.VideoCapture, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    return cap, frame_count, fps


def play_and_label(video_path: Path, clip_id: str, idx: int, total: int, current_status: str) -> str:
    cap, frame_count, fps = open_video_info(video_path)
    delay = max(1, int(1000 / fps))
    paused = False
    frame_idx = 0
    last_frame = None

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    continue
                last_frame = frame
                frame_idx += 1
            else:
                if last_frame is None:
                    ok, frame = cap.read()
                    if not ok:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        continue
                    last_frame = frame

            show = draw_overlay(
                last_frame,
                clip_id,
                idx,
                total,
                current_status if current_status else "unlabeled",
                paused,
            )
            cv2.imshow("quality_labeler", show)

            key = cv2.waitKey(0 if paused else delay) & 0xFF

            if key in (ord("a"), ord("1")):
                return "1"
            if key in (ord("d"), ord("0")):
                return "0"
            if key == ord("s"):
                return "skip"
            if key == ord("b"):
                return "back"
            if key == ord("q"):
                return "quit"
            if key == 32:
                paused = not paused
    finally:
        cap.release()


def label_dataset(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    clips_dir = dataset_root / "clips"
    meta_dir = dataset_root / "meta"
    clips_csv = meta_dir / "clips.csv"
    quality_csv = meta_dir / "quality_labels.csv"

    if not clips_csv.exists():
        raise FileNotFoundError(f"Missing clips.csv: {clips_csv}")
    if not clips_dir.exists():
        raise FileNotFoundError(f"Missing clips dir: {clips_dir}")

    clip_ids = load_clip_ids(clips_csv)
    if not clip_ids:
        raise RuntimeError("No clip_id found in clips.csv")

    quality_map = load_quality_map(quality_csv)
    rows = build_quality_rows(clip_ids, quality_map)

    if args.review_all:
        i = 0
    else:
        i = find_start_index(rows)

    total = len(rows)
    cv2.namedWindow("quality_labeler", cv2.WINDOW_NORMAL)

    while 0 <= i < total:
        clip_id = rows[i]["clip_id"]
        video_path = clips_dir / f"{clip_id}.mp4"
        if not video_path.exists():
            print(f"[skip] missing video: {video_path}")
            i += 1
            continue

        cur = rows[i]["accept"]
        if cur == "1":
            status = "accept"
        elif cur == "0":
            status = "reject"
        else:
            status = ""

        action = play_and_label(video_path, clip_id, i, total, status)

        if action == "1":
            rows[i]["accept"] = "1"
            rows[i]["reason"] = ""
            write_quality_csv(quality_csv, rows)
            print(f"[{i + 1}/{total}] {clip_id} -> accept")
            i += 1
            continue

        if action == "0":
            rows[i]["accept"] = "0"
            rows[i]["reason"] = ""
            write_quality_csv(quality_csv, rows)
            print(f"[{i + 1}/{total}] {clip_id} -> reject")
            i += 1
            continue

        if action == "skip":
            print(f"[{i + 1}/{total}] {clip_id} -> skip")
            i += 1
            continue

        if action == "back":
            i = max(0, i - 1)
            continue

        if action == "quit":
            write_quality_csv(quality_csv, rows)
            print("Saved and quit.")
            break

    cv2.destroyAllWindows()
    write_quality_csv(quality_csv, rows)

    labeled = sum(1 for row in rows if row["accept"] in {"0", "1"})
    accept_n = sum(1 for row in rows if row["accept"] == "1")
    reject_n = sum(1 for row in rows if row["accept"] == "0")

    print(f"Finished. labeled={labeled}/{total}, accept={accept_n}, reject={reject_n}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Keyboard-based quality labeler for tennis clips.")
    parser.add_argument(
        "--dataset-root",
        default="inputs/tennis_custom",
        help="Dataset root containing clips/ and meta/ (default: inputs/tennis_custom)",
    )
    parser.add_argument(
        "--review-all",
        action="store_true",
        help="Review from first clip, not from first unlabeled clip",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    label_dataset(args)


if __name__ == "__main__":
    main()