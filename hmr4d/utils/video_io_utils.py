import imageio.v3 as iio
import numpy as np
import torch
from pathlib import Path
import shutil
import ffmpeg
from tqdm import tqdm
import cv2


def get_video_lwh(video_path):
    L, H, W, _ = iio.improps(video_path, plugin="pyav").shape
    return L, W, H


def read_video_np(video_path, start_frame=0, end_frame=-1, scale=1.0):
    """
    Args:
        video_path: str
    Returns:
        frames: np.array, (N, H, W, 3) RGB, uint8
    """
    # If video path not exists, an error will be raised by ffmpegs
    filter_args = []
    should_check_length = False

    # 1. Trim
    if not (start_frame == 0 and end_frame == -1):
        if end_frame == -1:
            filter_args.append(("trim", f"start_frame={start_frame}"))
        else:
            should_check_length = True
            filter_args.append(("trim", f"start_frame={start_frame}:end_frame={end_frame}"))

    # 2. Scale
    if scale != 1.0:
        filter_args.append(("scale", f"iw*{scale}:ih*{scale}"))

    # Excute then check
    frames = iio.imread(video_path, plugin="pyav", filter_sequence=filter_args)
    if should_check_length:
        assert len(frames) == end_frame - start_frame

    return frames


def get_video_reader(video_path):
    return iio.imiter(video_path, plugin="pyav")


def read_images_np(image_paths, verbose=False):
    """
    Args:
        image_paths: list of str
    Returns:
        images: np.array, (N, H, W, 3) RGB, uint8
    """
    if verbose:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in tqdm(image_paths)]
    else:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in image_paths]
    images = np.stack(images, axis=0)
    return images


def save_video(images, video_path, fps=30, crf=17):
    """
    Args:
        images: (N, H, W, 3) RGB, uint8
        crf: 17 is visually lossless, 23 is default, +6 results in half the bitrate
    0 is lossless, https://trac.ffmpeg.org/wiki/Encode/H.264#crf
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().astype(np.uint8)
    elif isinstance(images, list):
        images = np.array(images).astype(np.uint8)

    with iio.imopen(video_path, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)
        writer._video_stream.options = {"crf": str(crf)}
        writer.write(images)


def get_writer(video_path, fps=30, crf=17):
    """remember to .close()"""
    writer = iio.imopen(video_path, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=fps)
    writer._video_stream.options = {"crf": str(crf)}
    return writer


def copy_file(video_path, out_video_path, overwrite=True):
    if not overwrite and Path(out_video_path).exists():
        return
    shutil.copy(video_path, out_video_path)


def merge_videos_horizontal(in_video_paths: list, out_video_path: str):
    if len(in_video_paths) < 2:
        raise ValueError("At least two video paths are required for merging.")
    inputs = [ffmpeg.input(path) for path in in_video_paths]
    merged_video = ffmpeg.filter(inputs, "hstack", inputs=len(inputs))
    output = ffmpeg.output(merged_video, out_video_path)
    ffmpeg.run(output, overwrite_output=True, quiet=True)


def merge_videos_vertical(in_video_paths: list, out_video_path: str):
    if len(in_video_paths) < 2:
        raise ValueError("At least two video paths are required for merging.")
    inputs = [ffmpeg.input(path) for path in in_video_paths]
    merged_video = ffmpeg.filter(inputs, "vstack", inputs=len(inputs))
    output = ffmpeg.output(merged_video, out_video_path)
    ffmpeg.run(output, overwrite_output=True, quiet=True)
