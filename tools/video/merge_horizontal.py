import argparse
from hmr4d.utils.video_io_utils import merge_videos_horizontal


def parse_args():
    """python tools/video/merge_horizontal.py a.mp4 b.mp4 c.mp4 -o out.mp4"""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_videos", nargs="+", help="Input video paths")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output video path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_videos_horizontal(args.input_videos, args.output)
