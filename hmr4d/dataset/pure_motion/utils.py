import torch
import torch.nn.functional as F
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from einops import rearrange


def aa_to_r6d(x):
    return matrix_to_rotation_6d(axis_angle_to_matrix(x))


def r6d_to_aa(x):
    return matrix_to_axis_angle(rotation_6d_to_matrix(x))


def interpolate_smpl_params(smpl_params, tgt_len):
    """
    smpl_params['body_pose'] (L, 63)
    tgt_len: L->L'
    """
    betas = smpl_params["betas"]
    body_pose = smpl_params["body_pose"]
    global_orient = smpl_params["global_orient"]  # (L, 3)
    transl = smpl_params["transl"]  # (L, 3)

    # Interpolate
    body_pose = rearrange(aa_to_r6d(body_pose.reshape(-1, 21, 3)), "l j c -> c j l")
    body_pose = F.interpolate(body_pose, tgt_len, mode="linear", align_corners=True)
    body_pose = r6d_to_aa(rearrange(body_pose, "c j l -> l j c")).reshape(-1, 63)

    # although this should be the same as above, we do it for consistency
    betas = rearrange(betas, "l c -> c 1 l")
    betas = F.interpolate(betas, tgt_len, mode="linear", align_corners=True)
    betas = rearrange(betas, "c 1 l -> l c")

    global_orient = rearrange(aa_to_r6d(global_orient.reshape(-1, 1, 3)), "l j c -> c j l")
    global_orient = F.interpolate(global_orient, tgt_len, mode="linear", align_corners=True)
    global_orient = r6d_to_aa(rearrange(global_orient, "c j l -> l j c")).reshape(-1, 3)

    transl = rearrange(transl, "l c -> c 1 l")
    transl = F.interpolate(transl, tgt_len, mode="linear", align_corners=True)
    transl = rearrange(transl, "c 1 l -> l c")

    return {"body_pose": body_pose, "betas": betas, "global_orient": global_orient, "transl": transl}


def rotate_around_axis(global_orient, transl, axis="y"):
    """Global coordinate augmentation. Random rotation around y-axis"""
    angle = torch.rand(1) * 2 * torch.pi
    if axis == "y":
        aa = torch.tensor([0.0, angle, 0.0]).float().unsqueeze(0)
    rmat = axis_angle_to_matrix(aa)

    global_orient = matrix_to_axis_angle(rmat @ axis_angle_to_matrix(global_orient))
    transl = (rmat.squeeze(0) @ transl.T).T
    return global_orient, transl


def augment_betas(betas, std=0.1):
    noise = torch.normal(mean=torch.zeros(10), std=torch.ones(10) * std)
    betas_aug = betas + noise[None]
    return betas_aug
