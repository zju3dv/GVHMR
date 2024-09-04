# Sebastian IK
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
import hmr4d.utils.matrix as matrix
from hmr4d.utils.geo.quaternion import qbetween, qslerp, qinv, qmul, qrot


class CCD_IK:
    def __init__(
        self,
        local_mat,
        parent,
        target_ind,
        target_pos=None,
        target_rot=None,
        kinematic_chain=None,
        max_iter=2,  # sebas sets 25 but with converged flag, 2 is enough
        threshold=0.001,
        pos_weight=1.0,
        rot_weight=0.0,  # this makes optimization unstable, although sebas uses 1.0
    ):
        if kinematic_chain is None:
            kinematic_chain = range(local_mat.shape[-3])
        global_mat = matrix.forward_kinematics(local_mat, parent)

        # get kinematic chain only local mat and assign root mat (do not modify root during IK)
        local_mat = local_mat.clone()
        local_mat = local_mat[..., kinematic_chain, :, :]
        local_mat[..., 0, :, :] = global_mat[..., kinematic_chain[0], :, :]

        parent = [i - 1 for i in range(len(kinematic_chain))]
        self.local_mat = local_mat
        self.global_mat = matrix.forward_kinematics(local_mat, parent)  # (*, J, 4, 4)
        self.parent = parent

        self.target_ind = target_ind
        if target_pos is not None:
            self.target_pos = target_pos  # (*, O, 3)
        else:
            self.target_pos = None
        if target_rot is not None:
            self.target_q = matrix_to_quaternion(target_rot)  # (*, O, 4)
        else:
            self.target_q = None

        self.threshold = threshold
        self.J_N = self.local_mat.shape[-3]
        self.target_N = len(target_ind)
        self.max_iter = max_iter
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight

    def is_converged(self):
        end_pos = matrix.get_position(self.global_mat)[..., self.target_ind, :]  # (*, OJ, 3)
        converged_mask = (self.target_pos - end_pos).norm(dim=-1) < self.threshold
        self.converged_mask = converged_mask
        if self.converged_mask.sum() > 0:
            return False
        return True

    def solve(self):
        for _ in range(self.max_iter):
            # if self.is_converged():
            #     return self.local_mat
            # do not optimize root, so start from 1
            self.optimize(1)
        return self.local_mat

    def optimize(self, i):
        # i: joint_i
        if i == self.J_N - 1:
            return
        pos = matrix.get_position(self.global_mat)[..., i, :]  # (*, 3)
        rot = matrix.get_rotation(self.global_mat)[..., i, :, :]  # (*, 3, 3)
        quat = matrix_to_quaternion(rot)  # (*, 4)
        x_vec = torch.zeros((quat.shape[:-1] + (3,)), device=quat.device)
        x_vec[..., 0] = 1.0
        x_vec_sum = torch.zeros_like(x_vec)
        y_vec = torch.zeros((quat.shape[:-1] + (3,)), device=quat.device)
        y_vec[..., 1] = 1.0
        y_vec_sum = torch.zeros_like(y_vec)

        count = 0

        for target_i, j in enumerate(self.target_ind):
            if i >= j:
                # do not optimise same joint or child joint of targets
                continue
            end_pos = matrix.get_position(self.global_mat)[..., j, :]  # (*, 3)
            end_rot = matrix.get_rotation(self.global_mat)[..., j, :, :]  # (*, 3, 3)
            end_quat = matrix_to_quaternion(end_rot)  # (*, 4)

            if self.target_pos is not None:
                target_pos = self.target_pos[..., target_i, :]  # (*, 3)
                # Solve objective position
                solved_pos_target_quat = qslerp(
                    quat,
                    qmul(qbetween(end_pos - pos, target_pos - pos), quat),
                    self.get_weight(i),
                )

                x_vec_sum += qrot(solved_pos_target_quat, x_vec)
                y_vec_sum += qrot(solved_pos_target_quat, y_vec)
                if self.pos_weight > 0:
                    count += 1

            if self.target_q is not None:
                if target_i < self.target_N - 1:
                    # multiple rot target makes more unstable, only keep the last one
                    continue
                # optimize rotation target is not stable
                target_q = self.target_q[..., target_i, :]  # (*, 4)
                # Solve objective rotation
                solved_q_target_quat = qslerp(
                    quat,
                    qmul(qmul(target_q, qinv(end_quat)), quat),
                    self.get_weight(i),
                )
                x_vec_sum += qrot(solved_q_target_quat, x_vec) * self.rot_weight
                y_vec_sum += qrot(solved_q_target_quat, y_vec) * self.rot_weight
                if self.rot_weight > 0:
                    count += 1

        if count > 0:
            x_vec_avg = matrix.normalize(x_vec_sum / count)
            y_vec_avg = matrix.normalize(y_vec_sum / count)
            z_vec_avg = torch.cross(x_vec_avg, y_vec_avg, dim=-1)
            solved_rot = torch.stack([x_vec_avg, y_vec_avg, z_vec_avg], dim=-1)  # column

            parent_rot = matrix.get_rotation(self.global_mat)[..., self.parent[i], :, :]
            solved_local_rot = matrix.get_mat_BtoA(parent_rot, solved_rot)
            self.local_mat[..., i, :-1, :-1] = solved_local_rot
            self.global_mat = matrix.forward_kinematics(self.local_mat, self.parent)
        self.optimize(i + 1)

    def get_weight(self, i):
        weight = (i + 1) / self.J_N
        return weight
