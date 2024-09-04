import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion, quaternion_to_matrix
import hmr4d.utils.matrix as matrix
from hmr4d.utils.net_utils import gaussian_smooth


def get_R_c2gv(R_w2c, axis_gravity_in_w=[0, 0, -1]):
    """
    Args:
        R_w2c: (*, 3, 3)
    Returns:
        R_c2gv: (*, 3, 3)
    """
    if isinstance(axis_gravity_in_w, list):
        axis_gravity_in_w = torch.tensor(axis_gravity_in_w).float()  # gravity direction in world coord
    axis_z_in_c = torch.tensor([0, 0, 1]).float()

    # get gv-coord axes in in c-coord
    axis_y_of_gv = R_w2c @ axis_gravity_in_w  # (*, 3)
    axis_x_of_gv = axis_y_of_gv.cross(axis_z_in_c.expand_as(axis_y_of_gv), dim=-1)
    # normalize
    axis_x_of_gv_norm = axis_x_of_gv.norm(dim=-1, keepdim=True)
    axis_x_of_gv = axis_x_of_gv / (axis_x_of_gv_norm + 1e-5)
    axis_x_of_gv[axis_x_of_gv_norm.squeeze(-1) < 1e-5] = torch.tensor([1.0, 0.0, 0.0])  # use cam x-axis as axis_x_of_gv
    axis_z_of_gv = axis_x_of_gv.cross(axis_y_of_gv, dim=-1)

    R_gv2c = torch.stack([axis_x_of_gv, axis_y_of_gv, axis_z_of_gv], dim=-1)  # (*, 3, 3)
    R_c2gv = R_gv2c.transpose(-1, -2)  # (*, 3, 3)
    return R_c2gv


tsf_axisangle = {
    "ay->ay": [0, 0, 0],
    "any->ay": [0, 0, torch.pi],
    "az->ay": [-torch.pi / 2, 0, 0],
    "ay->any": [0, 0, torch.pi],
}


def get_tgtcoord_rootparam(global_orient, transl, gravity_vec=None, tgt_gravity_vec=None, tsf="ay->ay"):
    """Rotate around the origin center, to match the new gravity direction
    Args:
        global_orient: torch.tensor, (*, 3)
        transl: torch.tensor, (*, 3)
        gravity_vec: torch.tensor, (3,)
        tgt_gravity_vec: torch.tensor, (3,)
    Returns:
        tgt_global_orient: torch.tensor, (*, 3)
        tgt_transl: torch.tensor, (*, 3)
        R_g2tg: (3, 3)
    """
    # get rotation matrix
    device = global_orient.device
    if gravity_vec is None and tgt_gravity_vec is None:
        aa = torch.tensor(tsf_axisangle[tsf]).to(device)
        R_g2tg = axis_angle_to_matrix(aa)  # (3, 3)
    else:
        raise NotImplementedError
        # TODO: Impl this function
        gravity_vec = torch.tensor(gravity_vec).float().to(device)
        gravity_vec = gravity_vec / gravity_vec.norm()
        tgt_gravity_vec = torch.tensor(tgt_gravity_vec).float().to(device)
        tgt_gravity_vec = tgt_gravity_vec / tgt_gravity_vec.norm()
        # pick one identity axis
        axis_identity = torch.tensor([0, 0, 0]).float().to(device)
        for i in (gravity_vec == 0) & (tgt_global_orient == 0):
            if i:
                axis_identity[i] = 1
                break

    # rotate
    global_orient_R = axis_angle_to_matrix(global_orient)  # (*, 3, 3)
    tgt_global_orient = matrix_to_axis_angle(R_g2tg @ global_orient_R)  # (*, 3, 3)
    tgt_transl = torch.einsum("...ij,...j->...i", R_g2tg, transl)

    return tgt_global_orient, tgt_transl, R_g2tg


def get_c_rootparam(global_orient, transl, T_w2c, offset):
    """
    Args:
        global_orient: torch.tensor, (F, 3)
        transl: torch.tensor, (F, 3)
        T_w2c: torch.tensor, (*, 4, 4)
        offset: torch.tensor, (3,)
    Returns:
        R_c: torch.tensor, (F, 3)
        t_c: torch.tensor, (F, 3)
    """
    assert global_orient.shape == transl.shape and len(global_orient.shape) == 2
    R_w = axis_angle_to_matrix(global_orient)  # (F, 3, 3)
    t_w = transl  # (F, 3)

    R_w2c = T_w2c[..., :3, :3]  # (*, 3, 3)
    t_w2c = T_w2c[..., :3, 3]  # (*, 3)
    if len(R_w2c.shape) == 2:
        R_w2c = R_w2c[None].expand(R_w.size(0), -1, -1)  # (F, 3, 3)
        t_w2c = t_w2c[None].expand(t_w.size(0), -1)

    R_c = matrix_to_axis_angle(R_w2c @ R_w)  # (F, 3)
    t_c = torch.einsum("fij,fj->fi", R_w2c, t_w + offset) + t_w2c - offset  # (F, 3)
    return R_c, t_c


def get_T_w2c_from_wcparams(global_orient_w, transl_w, global_orient_c, transl_c, offset):
    """
    Args:
        global_orient_w: torch.tensor, (F, 3)
        transl_w: torch.tensor, (F, 3)
        global_orient_c: torch.tensor, (F, 3)
        transl_c: torch.tensor, (F, 3)
        offset: torch.tensor, (*, 3)
    Returns:
        T_w2c: torch.tensor, (F, 4, 4)
    """
    assert global_orient_w.shape == transl_w.shape and len(global_orient_w.shape) == 2
    assert global_orient_c.shape == transl_c.shape and len(global_orient_c.shape) == 2

    R_w = axis_angle_to_matrix(global_orient_w)  # (F, 3, 3)
    t_w = transl_w  # (F, 3)
    R_c = axis_angle_to_matrix(global_orient_c)  # (F, 3, 3)
    t_c = transl_c  # (F, 3)

    R_w2c = R_c @ R_w.transpose(-1, -2)  # (F, 3, 3)
    t_w2c = t_c + offset - torch.einsum("fij,fj->fi", R_w2c, t_w + offset)  # (F, 3)
    T_w2c = torch.eye(4, device=global_orient_w.device).repeat(R_w.size(0), 1, 1)  # (F, 4, 4)
    T_w2c[..., :3, :3] = R_w2c  # (F, 3, 3)
    T_w2c[..., :3, 3] = t_w2c  # (F, 3)
    return T_w2c


def get_local_transl_vel(transl, global_orient):
    """
    transl velocity is in local coordinate (or, SMPL-coord)
    Args:
        transl: (*, L, 3)
        global_orient: (*, L, 3)
    Returns:
        transl_vel: (*, L, 3)
    """
    assert len(transl.shape) == len(global_orient.shape)
    global_orient_R = axis_angle_to_matrix(global_orient)  # (B, L, 3, 3)
    transl_vel = transl[..., 1:, :] - transl[..., :-1, :]  # (B, L-1, 3)
    transl_vel = torch.cat([transl_vel, transl_vel[..., [-1], :]], dim=-2)  # (B, L, 3)  last-padding

    # v_local = R^T @ v_global
    local_transl_vel = torch.einsum("...lij,...li->...lj", global_orient_R, transl_vel)
    return local_transl_vel


def rollout_local_transl_vel(local_transl_vel, global_orient, transl_0=None):
    """
    transl velocity is in local coordinate (or, SMPL-coord)
    Args:
        local_transl_vel: (*, L, 3)
        global_orient: (*, L, 3)
        transl_0: (*, 1, 3), if not provided, the start point is 0
    Returns:
        transl: (*, L, 3)
    """
    global_orient_R = axis_angle_to_matrix(global_orient)
    transl_vel = torch.einsum("...lij,...lj->...li", global_orient_R, local_transl_vel)

    # set start point
    if transl_0 is None:
        transl_0 = transl_vel[..., :1, :].clone().detach().zero_()
    transl_ = torch.cat([transl_0, transl_vel[..., :-1, :]], dim=-2)

    # rollout from start point
    transl = torch.cumsum(transl_, dim=-2)
    return transl


def get_local_transl_vel_alignhead(transl, global_orient):
    # assume global_orient is ay
    global_orient_rot = axis_angle_to_matrix(global_orient)  # (*, 3, 3)
    global_orient_quat = matrix_to_quaternion(global_orient_rot)  # (*, 4)

    global_orient_quat_xyzw = matrix.quat_wxyz2xyzw(global_orient_quat)  # (*, 4)
    head_quat_xyzw = matrix.calc_heading_quat(global_orient_quat_xyzw, head_ind=2, gravity_axis="y")  # (*, 4)
    head_quat = matrix.quat_xyzw2wxyz(head_quat_xyzw)  # (*, 4)
    head_rot = quaternion_to_matrix(head_quat)
    head_aa = matrix_to_axis_angle(head_rot)

    local_transl_vel_alignhead = get_local_transl_vel(transl, head_aa)
    return local_transl_vel_alignhead


def rollout_local_transl_vel_alignhead(local_transl_vel_alignhead, global_orient, transl_0=None):
    # assume global_orient is ay
    global_orient_rot = axis_angle_to_matrix(global_orient)  # (*, 3, 3)
    global_orient_quat = matrix_to_quaternion(global_orient_rot)  # (*, 4)

    global_orient_quat_xyzw = matrix.quat_wxyz2xyzw(global_orient_quat)  # (*, 4)
    head_quat_xyzw = matrix.calc_heading_quat(global_orient_quat_xyzw, head_ind=2, gravity_axis="y")  # (*, 4)
    head_quat = matrix.quat_xyzw2wxyz(head_quat_xyzw)  # (*, 4)
    head_rot = quaternion_to_matrix(head_quat)
    head_aa = matrix_to_axis_angle(head_rot)

    transl = rollout_local_transl_vel(local_transl_vel_alignhead, head_aa, transl_0)
    return transl


def get_local_transl_vel_alignhead_absy(transl, global_orient):
    # assume global_orient is ay
    global_orient_rot = axis_angle_to_matrix(global_orient)  # (*, 3, 3)
    global_orient_quat = matrix_to_quaternion(global_orient_rot)  # (*, 4)

    global_orient_quat_xyzw = matrix.quat_wxyz2xyzw(global_orient_quat)  # (*, 4)
    head_quat_xyzw = matrix.calc_heading_quat(global_orient_quat_xyzw, head_ind=2, gravity_axis="y")  # (*, 4)
    head_quat = matrix.quat_xyzw2wxyz(head_quat_xyzw)  # (*, 4)
    head_rot = quaternion_to_matrix(head_quat)
    head_aa = matrix_to_axis_angle(head_rot)

    local_transl_vel_alignhead = get_local_transl_vel(transl, head_aa)
    abs_y = torch.cumsum(local_transl_vel_alignhead[..., [1]], dim=-2)  # (*, L, 1)
    local_transl_vel_alignhead_absy = torch.cat(
        [local_transl_vel_alignhead[..., [0]], abs_y, local_transl_vel_alignhead[..., [2]]], dim=-1
    )

    return local_transl_vel_alignhead_absy


def rollout_local_transl_vel_alignhead_absy(local_transl_vel_alignhead_absy, global_orient, transl_0=None):
    # assume global_orient is ay
    global_orient_rot = axis_angle_to_matrix(global_orient)  # (*, 3, 3)
    global_orient_quat = matrix_to_quaternion(global_orient_rot)  # (*, 4)

    global_orient_quat_xyzw = matrix.quat_wxyz2xyzw(global_orient_quat)  # (*, 4)
    head_quat_xyzw = matrix.calc_heading_quat(global_orient_quat_xyzw, head_ind=2, gravity_axis="y")  # (*, 4)
    head_quat = matrix.quat_xyzw2wxyz(head_quat_xyzw)  # (*, 4)
    head_rot = quaternion_to_matrix(head_quat)
    head_aa = matrix_to_axis_angle(head_rot)

    local_transl_vel_alignhead_y = (
        local_transl_vel_alignhead_absy[..., 1:, [1]] - local_transl_vel_alignhead_absy[..., :-1, [1]]
    )
    local_transl_vel_alignhead_y = torch.cat(
        [local_transl_vel_alignhead_absy[..., :1, [1]], local_transl_vel_alignhead_y], dim=-2
    )
    local_transl_vel_alignhead = torch.cat(
        [
            local_transl_vel_alignhead_absy[..., [0]],
            local_transl_vel_alignhead_y,
            local_transl_vel_alignhead_absy[..., [2]],
        ],
        dim=-1,
    )

    transl = rollout_local_transl_vel(local_transl_vel_alignhead, head_aa, transl_0)
    return transl


def get_local_transl_vel_alignhead_absgy(transl, global_orient):
    # assume global_orient is ay
    global_orient_rot = axis_angle_to_matrix(global_orient)  # (*, 3, 3)
    global_orient_quat = matrix_to_quaternion(global_orient_rot)  # (*, 4)

    global_orient_quat_xyzw = matrix.quat_wxyz2xyzw(global_orient_quat)  # (*, 4)
    head_quat_xyzw = matrix.calc_heading_quat(global_orient_quat_xyzw, head_ind=2, gravity_axis="y")  # (*, 4)
    head_quat = matrix.quat_xyzw2wxyz(head_quat_xyzw)  # (*, 4)
    head_rot = quaternion_to_matrix(head_quat)
    head_aa = matrix_to_axis_angle(head_rot)

    local_transl_vel_alignhead = get_local_transl_vel(transl, head_aa)
    abs_y = transl[..., [1]]  # (*, L, 1)
    local_transl_vel_alignhead_absy = torch.cat(
        [local_transl_vel_alignhead[..., [0]], abs_y, local_transl_vel_alignhead[..., [2]]], dim=-1
    )

    return local_transl_vel_alignhead_absy


def rollout_local_transl_vel_alignhead_absgy(local_transl_vel_alignhead_absgy, global_orient, transl_0=None):
    # assume global_orient is ay
    global_orient_rot = axis_angle_to_matrix(global_orient)  # (*, 3, 3)
    global_orient_quat = matrix_to_quaternion(global_orient_rot)  # (*, 4)

    global_orient_quat_xyzw = matrix.quat_wxyz2xyzw(global_orient_quat)  # (*, 4)
    head_quat_xyzw = matrix.calc_heading_quat(global_orient_quat_xyzw, head_ind=2, gravity_axis="y")  # (*, 4)
    head_quat = matrix.quat_xyzw2wxyz(head_quat_xyzw)  # (*, 4)
    head_rot = quaternion_to_matrix(head_quat)
    head_aa = matrix_to_axis_angle(head_rot)

    local_transl_vel_alignhead_y = (
        local_transl_vel_alignhead_absgy[..., 1:, [1]] - local_transl_vel_alignhead_absgy[..., :-1, [1]]
    )
    local_transl_vel_alignhead_y = torch.cat(
        [local_transl_vel_alignhead_y, local_transl_vel_alignhead_y[..., -1:, :]], dim=-2
    )
    if transl_0 is not None:
        transl_0 = transl_0.clone()
        transl_0[..., 1] = local_transl_vel_alignhead_absgy[..., :1, 1]
    else:
        transl_0 = local_transl_vel_alignhead_absgy.clone()[..., :1, :]  # (*, 1, 3)
        transl_0[..., :1, 0] = 0.0
        transl_0[..., :1, 2] = 0.0

    local_transl_vel_alignhead = torch.cat(
        [
            local_transl_vel_alignhead_absgy[..., [0]],
            local_transl_vel_alignhead_y,
            local_transl_vel_alignhead_absgy[..., [2]],
        ],
        dim=-1,
    )

    transl = rollout_local_transl_vel(local_transl_vel_alignhead, head_aa, transl_0)
    return transl


def rollout_vel(vel, transl_0=None):
    """
    Args:
        vel: (*, L, 3)
        transl_0: (*, 1, 3), if not provided, the start point is 0
    Returns:
        transl: (*, L, 3)
    """
    # set start point
    if transl_0 is None:
        assert len(vel.shape) == len(transl_0.shape)
        transl_0 = vel[..., :1, :].clone().detach().zero_()
    transl_ = torch.cat([transl_0, vel[..., :-1, :]], dim=-2)

    # rollout from start point
    transl = torch.cumsum(transl_, dim=-2)
    return transl


def get_static_joint_mask(w_j3d, vel_thr=0.25, smooth=False, repeat_last=False):
    """
    w_j3d: (*, L, J, 3)
    vel_thr: HuMoR uses 0.15m/s
    """
    joint_v_ = (w_j3d[..., 1:, :, :] - w_j3d[..., :-1, :, :]).pow(2).sum(-1).sqrt() / 0.033  # (*, L-1, J)
    if smooth:
        joint_v_ = gaussian_smooth(joint_v_, 3, -2)

    static_joint_mask = joint_v_ < vel_thr  # 1 as stable, 0 as moving

    if repeat_last:  # repeat the last frame, this makes the shape same as w_j3d
        static_joint_mask = torch.cat([static_joint_mask, static_joint_mask[..., [-1], :]], dim=-2)

    return static_joint_mask
