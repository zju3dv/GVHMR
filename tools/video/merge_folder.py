"""This script will glob two folder, check the mp4 files are one-to-one match precisely, then call merge_horizontal.py to merge them one by one"""

import os
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir1", type=str)
    parser.add_argument("input_dir2", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--vertical", action="store_true")  # By default use horizontal
    args = parser.parse_args()

    # Check input
    input_dir1 = Path(args.input_dir1)
    input_dir2 = Path(args.input_dir2)
    assert input_dir1.exists()
    assert input_dir2.exists()
    video_paths1 = sorted(input_dir1.glob("*.mp4"))
    video_paths2 = sorted(input_dir2.glob("*.mp4"))
    assert len(video_paths1) == len(video_paths2)
    for path1, path2 in zip(video_paths1, video_paths2):
        assert path1.stem == path2.stem

    # Merge to output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path1, path2 in zip(video_paths1, video_paths2):
        out_path = output_dir / f"{path1.stem}.mp4"
        in_paths = [str(path1), str(path2)]
        print(f"Merging {in_paths} to {out_path}")
        if args.vertical:
            os.system(f"python tools/video/merge_vertical.py {' '.join(in_paths)} -o {out_path}")
        else:
            os.system(f"python tools/video/merge_horizontal.py {' '.join(in_paths)} -o {out_path}")


if __name__ == "__main__":
    main()
