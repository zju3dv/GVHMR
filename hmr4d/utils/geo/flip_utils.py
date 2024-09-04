import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


def flip_heatmap_coco17(output_flipped):
    assert output_flipped.ndim == 4, "output_flipped should be [B, J, H, W]"
    shape_ori = output_flipped.shape
    channels = 1
    output_flipped = output_flipped.reshape(shape_ori[0], -1, channels, shape_ori[2], shape_ori[3])
    output_flipped_back = output_flipped.clone()

    # Swap left-right parts
    for left, right in [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]:
        output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
        output_flipped_back[:, right, ...] = output_flipped[:, left, ...]
    output_flipped_back = output_flipped_back.reshape(shape_ori)
    # Flip horizontally
    output_flipped_back = output_flipped_back.flip(3)
    return output_flipped_back


def flip_bbx_xys(bbx_xys, w):
    """
    bbx_xys: (F, 3)
    """
    bbx_xys_flip = bbx_xys.clone()
    bbx_xys_flip[:, 0] = w - bbx_xys_flip[:, 0]
    return bbx_xys_flip


def flip_kp2d_coco17(kp2d, w):
    """Flip keypoints."""
    kp2d = kp2d.clone()
    flipped_parts = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    kp2d = kp2d[..., flipped_parts, :]
    kp2d[..., 0] = w - kp2d[..., 0]
    return kp2d


def flip_smplx_params(smplx_params):
    """Flip pose.
    The flipping is based on SMPLX parameters.
    """
    rotation = torch.cat([smplx_params["global_orient"], smplx_params["body_pose"]], dim=1)

    BN = rotation.shape[0]
    pose = rotation.reshape(BN, -1).transpose(0, 1)

    SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]  # , 23, 22]
    SMPL_POSE_FLIP_PERM = []
    for i in SMPL_JOINTS_FLIP_PERM:
        SMPL_POSE_FLIP_PERM.append(3 * i)
        SMPL_POSE_FLIP_PERM.append(3 * i + 1)
        SMPL_POSE_FLIP_PERM.append(3 * i + 2)

    pose = pose[SMPL_POSE_FLIP_PERM]

    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    pose = pose.transpose(0, 1).reshape(BN, -1, 3)

    smplx_params_flipped = smplx_params.copy()
    smplx_params_flipped["global_orient"] = pose[:, :1]
    smplx_params_flipped["body_pose"] = pose[:, 1:]
    return smplx_params_flipped


def avg_smplx_aa(aa1, aa2):
    def avg_rot(rot):
        # input [B,...,3,3] --> output [...,3,3]
        rot = rot.mean(dim=0)
        U, _, V = torch.svd(rot)
        rot = U @ V.transpose(-1, -2)
        return rot

    B, J3 = aa1.shape
    aa1 = aa1.reshape(B, -1, 3)
    aa2 = aa2.reshape(B, -1, 3)

    R1 = axis_angle_to_matrix(aa1)
    R2 = axis_angle_to_matrix(aa2)
    R_avg = avg_rot(torch.stack([R1, R2]))
    aa_avg = matrix_to_axis_angle(R_avg).reshape(B, -1)

    return aa_avg
