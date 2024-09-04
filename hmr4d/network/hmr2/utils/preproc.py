import cv2
import numpy as np
import torch
from pathlib import Path

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225])


def expand_to_aspect_ratio(input_shape, target_aspect_ratio=[192, 256]):
    """Increase the size of the bounding box to match the target shape."""
    if target_aspect_ratio is None:
        return input_shape

    try:
        w, h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    if h_new < h or w_new < w:
        breakpoint()
    return np.array([w_new, h_new])


def crop_and_resize(img, bbx_xy, bbx_s, dst_size=256, enlarge_ratio=1.2):
    """
    Args:
        img: (H, W, 3)
        bbx_xy: (2,)
        bbx_s: scalar
    """
    hs = bbx_s * enlarge_ratio / 2
    src = np.stack(
        [
            bbx_xy - hs,  # left-up corner
            bbx_xy + np.array([hs, -hs]),  # right-up corner
            bbx_xy,  # center
        ]
    ).astype(np.float32)
    dst = np.array([[0, 0], [dst_size - 1, 0], [dst_size / 2 - 0.5, dst_size / 2 - 0.5]], dtype=np.float32)
    A = cv2.getAffineTransform(src, dst)

    img_crop = cv2.warpAffine(img, A, (dst_size, dst_size), flags=cv2.INTER_LINEAR)
    bbx_xys_final = np.array([*bbx_xy, bbx_s * enlarge_ratio])
    return img_crop, bbx_xys_final
