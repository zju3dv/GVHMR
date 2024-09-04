import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
import hmr4d.utils.matrix as matrix
from hmr4d import PROJ_ROOT

COCO17_AUG = {k: v.flatten() for k, v in torch.load(PROJ_ROOT / "hmr4d/utils/body_model/coco_aug_dict.pth").items()}
COCO17_AUG_CUDA = {}
COCO17_TREE = [[5, 6], 0, 0, 1, 2, -1, -1, 5, 6, 7, 8, -1, -1, 11, 12, 13, 14, 15, 15, 15, 16, 16, 16]


def gaussian_augment(body_pose, std_angle=10.0, to_R=True):
    """
    Args:
        body_pose torch.Tensor: (..., J, 3) axis-angle if to_R is True, else rotmat (..., J, 3, 3)
        std_angle: scalar or list, in degree
    """

    body_pose = body_pose.clone()

    if to_R:
        body_pose_R = axis_angle_to_matrix(body_pose)  # (B, L, J, 3, 3)
    else:
        body_pose_R = body_pose
    shape = body_pose_R.shape[:-2]
    device = body_pose.device

    # 1. Simulate noise
    # angle:
    std_angle = torch.tensor(std_angle).to(device).reshape(-1)  # allow scalar or list
    noise_angle = torch.randn(shape, device=device) * std_angle * torch.pi / 180

    # axis: avoid zero vector
    noise_axis = torch.rand((*shape, 3), device=device)
    mask_ = torch.norm(noise_axis, dim=-1) < 1e-6
    noise_axis[mask_] = 1

    noise_axis = noise_axis / torch.norm(noise_axis, dim=-1, keepdim=True)
    noise_aa = noise_angle[..., None] * noise_axis  # (B, L, J, 3)
    noise_R = axis_angle_to_matrix(noise_aa)  # (B, L, J, 3, 3)

    # 2. Add noise to body pose
    new_body_pose_R = matrix.get_mat_BfromA(body_pose_R, noise_R)  # (B, L, J, 3, 3)
    # new_body_pose_R = torch.matmul(noise_R, body_pose_R)
    new_body_pose_r6d = matrix_to_rotation_6d(new_body_pose_R)  # (B, L, J, 6)
    new_body_pose_aa = matrix_to_axis_angle(new_body_pose_R)  # (B, L, J, 3)

    return new_body_pose_R, new_body_pose_r6d, new_body_pose_aa


# ========= Augment Joint 3D ======== #


def get_jitter(shape=(8, 120), s_jittering=5e-2):
    """Guassian jitter modeling."""
    jittering_noise = (
        torch.normal(
            mean=torch.zeros((*shape, 17, 3)),
            std=COCO17_AUG["jittering"].reshape(1, 1, 17, 1).expand(*shape, -1, 3),
        )
        * s_jittering
    )
    return jittering_noise


def get_jitter_cuda(shape=(8, 120), s_jittering=5e-2):
    if "jittering" not in COCO17_AUG_CUDA:
        COCO17_AUG_CUDA["jittering"] = COCO17_AUG["jittering"].cuda().reshape(1, 1, 17, 1)
    jittering = COCO17_AUG_CUDA["jittering"]
    jittering_noise = torch.randn((*shape, 17, 3), device="cuda") * jittering * s_jittering
    return jittering_noise


def get_lfhp(shape=(8, 120), s_peak=3e-1, s_peak_mask=5e-3):
    """Low-frequency high-peak noise modeling."""

    def get_peak_noise_mask():
        peak_noise_mask = torch.rand(*shape, 17) * COCO17_AUG["pmask"]
        peak_noise_mask = peak_noise_mask < s_peak_mask
        return peak_noise_mask

    peak_noise_mask = get_peak_noise_mask()  # (B, L, 17)
    peak_noise = peak_noise_mask.float().unsqueeze(-1).repeat(1, 1, 1, 3)
    peak_noise = peak_noise * torch.randn(3) * COCO17_AUG["peak"].reshape(17, 1) * s_peak
    return peak_noise


def get_lfhp_cuda(shape=(8, 120), s_peak=3e-1, s_peak_mask=5e-3):
    if "peak" not in COCO17_AUG_CUDA:
        COCO17_AUG_CUDA["pmask"] = COCO17_AUG["pmask"].cuda()
        COCO17_AUG_CUDA["peak"] = COCO17_AUG["peak"].cuda().reshape(17, 1)

    pmask = COCO17_AUG_CUDA["pmask"]
    peak = COCO17_AUG_CUDA["peak"]
    peak_noise_mask = torch.rand(*shape, 17, device="cuda") * pmask < s_peak_mask
    peak_noise = (
        peak_noise_mask.float().unsqueeze(-1).expand(-1, -1, -1, 3) * torch.randn(3, device="cuda") * peak * s_peak
    )
    return peak_noise


def get_bias(shape=(8, 120), s_bias=1e-1):
    """Bias noise modeling."""
    b, l = shape
    bias_noise = torch.normal(mean=torch.zeros((b, 17, 3)), std=COCO17_AUG["bias"].reshape(1, 17, 1)) * s_bias
    bias_noise = bias_noise[:, None].expand(-1, l, -1, -1)  # (B, L, J, 3), the whole sequence is moved by the same bias
    return bias_noise


def get_bias_cuda(shape=(8, 120), s_bias=1e-1):
    if "bias" not in COCO17_AUG_CUDA:
        COCO17_AUG_CUDA["bias"] = COCO17_AUG["bias"].cuda().reshape(1, 17, 1)

    bias = COCO17_AUG_CUDA["bias"]
    bias_noise = torch.randn((shape[0], 17, 3), device="cuda") * bias * s_bias
    bias_noise = bias_noise[:, None].expand(-1, shape[1], -1, -1)
    return bias_noise


def get_wham_aug_kp3d(shape=(8, 120)):
    # aug = get_bias(shape).cuda() + get_lfhp(shape).cuda() + get_jitter(shape).cuda()
    aug = get_bias_cuda(shape) + get_lfhp_cuda(shape) + get_jitter_cuda(shape)
    return aug


def get_visible_mask(shape=(8, 120), s_mask=0.03):
    """Mask modeling."""
    # Per-frame and joint
    mask = torch.rand(*shape, 17) < s_mask
    visible = (~mask).clone()  # (B, L, 17)

    visible = visible.reshape(-1, 17)  # (BL, 17)
    for child in range(17):
        parent = COCO17_TREE[child]
        if parent == -1:
            continue
        if isinstance(parent, list):
            visible[:, child] *= visible[:, parent[0]] * visible[:, parent[1]]
        else:
            visible[:, child] *= visible[:, parent]
    visible = visible.reshape(*shape, 17).clone()  # (B, L, J)
    return visible


def get_invisible_legs_mask(shape, s_mask=0.03):
    """
    Both legs are invisible for a random duration.
    """
    B, L = shape
    starts = torch.randint(0, L - 90, (B,))
    ends = starts + torch.randint(30, 90, (B,))
    mask_range = torch.arange(L).unsqueeze(0).expand(B, -1)
    mask_to_apply = (mask_range >= starts.unsqueeze(1)) & (mask_range < ends.unsqueeze(1))
    mask_to_apply = mask_to_apply.unsqueeze(2).expand(-1, -1, 17).clone()
    mask_to_apply[:, :, :11] = False  # only both legs are invisible
    mask_to_apply = mask_to_apply & (torch.rand(B, 1, 1) < s_mask)
    return mask_to_apply


def randomly_occlude_lower_half(i_x2d, s_mask=0.03):
    """
    Randomly occlude the lower half of the image.
    """
    raise NotImplementedError
    B, L, N, _ = i_x2d.shape
    i_x2d = i_x2d.clone()

    # a period of time when the lower half of the image is invisible
    starts = torch.randint(0, L - 90, (B,))
    ends = starts + torch.randint(30, 90, (B,))
    mask_range = torch.arange(L).unsqueeze(0).expand(B, -1)
    mask_to_apply = (mask_range >= starts.unsqueeze(1)) & (mask_range < ends.unsqueeze(1))
    mask_to_apply = mask_to_apply.unsqueeze(2).expand(-1, -1, N)  # (B, L, N)

    # only the lower half of the image is invisible
    i_x2d
    i_x2d[..., 1] / 2

    mask_to_apply = mask_to_apply & (torch.rand(B, 1, 1) < s_mask)
    return mask_to_apply


def randomly_modify_hands_legs(j3d):
    hands = [9, 10]
    legs = [15, 16]

    B, L, J, _ = j3d.shape
    p_switch_hand = 0.001
    p_switch_leg = 0.001
    p_wrong_hand0 = 0.001
    p_wrong_hand1 = 0.001
    p_wrong_leg0 = 0.001
    p_wrong_leg1 = 0.001

    mask = torch.rand(B, L) < p_switch_hand
    j3d[mask][:, hands] = j3d[mask][:, hands[::-1]]
    mask = torch.rand(B, L) < p_switch_leg
    j3d[mask][:, legs] = j3d[mask][:, legs[::-1]]
    mask = torch.rand(B, L) < p_wrong_hand0
    j3d[mask][:, 9] = j3d[mask][:, 10]
    mask = torch.rand(B, L) < p_wrong_hand1
    j3d[mask][:, 10] = j3d[mask][:, 9]
    mask = torch.rand(B, L) < p_wrong_leg0
    j3d[mask][:, 15] = j3d[mask][:, 16]
    mask = torch.rand(B, L) < p_wrong_leg1
    j3d[mask][:, 16] = j3d[mask][:, 15]

    return j3d
