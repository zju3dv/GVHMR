import torch
import torch.nn.functional as F
import numpy as np
from .vitpose_pytorch import build_model
from .vitfeat_extractor import get_batch
from tqdm import tqdm

from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps
from hmr4d.utils.geo_transform import cvt_p2d_from_pm1_to_i
from hmr4d.utils.geo.flip_utils import flip_heatmap_coco17


class VitPoseExtractor:
    def __init__(self, tqdm_leave=True):
        ckpt_path = "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
        self.pose = build_model("ViTPose_huge_coco_256x192", ckpt_path)
        self.pose.cuda().eval()

        self.flip_test = True
        self.tqdm_leave = tqdm_leave

    @torch.no_grad()
    def extract(self, video_path, bbx_xys, img_ds=0.5):
        # Get the batch
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        L, _, H, W = imgs.shape  # (L, 3, H, W)
        batch_size = 16
        vitpose = []
        for j in tqdm(range(0, L, batch_size), desc="ViTPose", leave=self.tqdm_leave):
            # Heat map
            imgs_batch = imgs[j : j + batch_size, :, :, 32:224].cuda()
            if self.flip_test:
                heatmap, heatmap_flipped = self.pose(torch.cat([imgs_batch, imgs_batch.flip(3)], dim=0)).chunk(2)
                heatmap_flipped = flip_heatmap_coco17(heatmap_flipped)
                heatmap = (heatmap + heatmap_flipped) * 0.5
                del heatmap_flipped
            else:
                heatmap = self.pose(imgs_batch.clone())  # (B, J, 64, 48)

            if False:
                # Get joint
                bbx_xys_batch = bbx_xys[j : j + batch_size].cuda()
                method = "hard"
                if method == "hard":
                    kp2d_pm1, conf = get_heatmap_preds(heatmap)
                elif method == "soft":
                    kp2d_pm1, conf = get_heatmap_preds(heatmap, soft=True)

                # Convert 64, 48 to 64, 64
                kp2d_pm1[:, :, 0] *= 24 / 32
                kp2d = cvt_p2d_from_pm1_to_i(kp2d_pm1, bbx_xys_batch[:, None])
                kp2d = torch.cat([kp2d, conf], dim=-1)

            else:  # postprocess from mmpose
                bbx_xys_batch = bbx_xys[j : j + batch_size]
                heatmap = heatmap.clone().cpu().numpy()
                center = bbx_xys_batch[:, :2].numpy()
                scale = (torch.cat((bbx_xys_batch[:, [2]] * 24 / 32, bbx_xys_batch[:, [2]]), dim=1) / 200).numpy()
                preds, maxvals = keypoints_from_heatmaps(heatmaps=heatmap, center=center, scale=scale, use_udp=True)
                kp2d = np.concatenate((preds, maxvals), axis=-1)
                kp2d = torch.from_numpy(kp2d)

            vitpose.append(kp2d.detach().cpu().clone())

        vitpose = torch.cat(vitpose, dim=0).clone()  # (F, 17, 3)
        return vitpose


def get_heatmap_preds(heatmap, normalize_keypoints=True, thr=0.0, soft=False):
    """
    heatmap: (B, J, H, W)
    """
    assert heatmap.ndim == 4, "batch_images should be 4-ndim"

    B, J, H, W = heatmap.shape
    heatmaps_reshaped = heatmap.reshape((B, J, -1))

    maxvals, idx = torch.max(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape((B, J, 1))
    idx = idx.reshape((B, J, 1))
    preds = idx.repeat(1, 1, 2).float()
    preds[:, :, 0] = (preds[:, :, 0]) % W
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / W)

    pred_mask = torch.gt(maxvals, thr).repeat(1, 1, 2)
    pred_mask = pred_mask.float()
    preds *= pred_mask

    # soft peak
    if soft:
        patch_size = 5
        patch_half = patch_size // 2
        patches = torch.zeros((B, J, patch_size, patch_size)).to(heatmap)
        default_patch = torch.zeros(patch_size, patch_size).to(heatmap)
        default_patch[patch_half, patch_half] = 1
        for b in range(B):
            for j in range(17):
                x, y = preds[b, j].int()
                if x >= patch_half and x <= W - patch_half and y >= patch_half and y <= H - patch_half:
                    patches[b, j] = heatmap[
                        b, j, y - patch_half : y + patch_half + 1, x - patch_half : x + patch_half + 1
                    ]
                else:
                    patches[b, j] = default_patch

        dx, dy = soft_patch_dx_dy(patches)
        preds[:, :, 0] += dx
        preds[:, :, 1] += dy

    if normalize_keypoints:  # to [-1, 1]
        preds[:, :, 0] = preds[:, :, 0] / (W - 1) * 2 - 1
        preds[:, :, 1] = preds[:, :, 1] / (H - 1) * 2 - 1

    return preds, maxvals


def soft_patch_dx_dy(p):
    """p (B,J,P,P)"""
    p_batch_shape = p.shape[:-2]
    patch_size = p.size(-1)
    temperature = 1.0
    score = F.softmax(p.view(-1, patch_size**2) * temperature, dim=-1)

    # get a offset_grid (BN, P, P, 2) for dx, dy
    offset_grid = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size))[::-1]
    offset_grid = torch.stack(offset_grid, dim=-1).float() - (patch_size - 1) / 2
    offset_grid = offset_grid.view(1, 1, patch_size, patch_size, 2).to(p.device)

    score = score.view(*p_batch_shape, patch_size, patch_size)
    dx = torch.sum(score * offset_grid[..., 0], dim=(-2, -1))
    dy = torch.sum(score * offset_grid[..., 1], dim=(-2, -1))

    if False:
        b, j = 0, 0
        print(torch.stack([dx[b, j], dy[b, j]]))
        print(p[b, j])

    return dx, dy
