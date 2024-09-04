import torch
import torch.nn as nn
from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from hmr4d.configs import MainStore, builds
from hmr4d.utils.geo.augment_noisy_pose import gaussian_augment
import hmr4d.utils.matrix as matrix
from hmr4d.utils.pylogger import Log
from hmr4d.utils.geo.hmr_global import get_local_transl_vel, rollout_local_transl_vel
from hmr4d.utils.smplx_utils import make_smplx
from . import stats_compose


class EnDecoder(nn.Module):
    def __init__(self, stats_name="DEFAULT_01", noise_pose_k=10):
        super().__init__()
        # Load mean, std
        stats = getattr(stats_compose, stats_name)
        Log.info(f"[EnDecoder] Use {stats_name} for statistics!")
        self.register_buffer("mean", torch.tensor(stats["mean"]).float(), False)
        self.register_buffer("std", torch.tensor(stats["std"]).float(), False)

        # option
        self.noise_pose_k = noise_pose_k

        # smpl
        self.smplx_model = make_smplx("supermotion_v437coco17")
        parents = self.smplx_model.parents[:22]
        self.register_buffer("parents_tensor", parents, False)
        self.parents = parents.tolist()

    def get_noisyobs(self, data, return_type="r6d"):
        """
        Noisy observation contains local pose with noise
        Args:
            data (dict):
                body_pose: (B, L, J*3) or (B, L, J, 3)
        Returns:
            noisy_bosy_pose: (B, L, J, 6) or (B, L, J, 3) or (B, L, 3, 3) depends on return_type
        """
        body_pose = data["body_pose"]  # (B, L, 63)
        B, L, _ = body_pose.shape
        body_pose = body_pose.reshape(B, L, -1, 3)

        # (B, L, J, C)
        return_mapping = {"R": 0, "r6d": 1, "aa": 2}
        return_id = return_mapping[return_type]
        noisy_bosy_pose = gaussian_augment(body_pose, self.noise_pose_k, to_R=True)[return_id]
        return noisy_bosy_pose

    def normalize_body_pose_r6d(self, body_pose_r6d):
        """body_pose_r6d: (B, L, {J*6}/{J, 6}) ->  (B, L, J*6)"""
        B, L = body_pose_r6d.shape[:2]
        body_pose_r6d = body_pose_r6d.reshape(B, L, -1)
        if self.mean.shape[-1] == 1:  # no mean, std provided
            return body_pose_r6d
        body_pose_r6d = (body_pose_r6d - self.mean[:126]) / self.std[:126]  # (B, L, C)
        return body_pose_r6d

    def fk_v2(self, body_pose, betas, global_orient=None, transl=None, get_intermediate=False):
        """
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
        Returns:
            joints: (B, L, 22, 3)
        """
        B, L = body_pose.shape[:2]
        if global_orient is None:
            global_orient = torch.zeros((B, L, 3), device=body_pose.device)
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)

        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)

        if transl is not None:
            local_skeleton[..., 0, :] += transl  # B, L, 22, 3

        mat = matrix.get_TRS(rotmat, local_skeleton)  # B, L, 22, 4, 4
        fk_mat = matrix.forward_kinematics(mat, self.parents)  # B, L, 22, 4, 4
        joints = matrix.get_position(fk_mat)  # B, L, 22, 3
        if not get_intermediate:
            return joints
        else:
            return joints, mat, fk_mat

    def get_local_pos(self, betas):
        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)
        return local_skeleton

    def encode(self, inputs):
        """
        definition: {
                body_pose_r6d,  # (B, L, (J-1)*6) -> 0:126
                betas, # (B, L, 10) -> 126:136
                global_orient_r6d,  # (B, L, 6) -> 136:142  incam
                global_orient_gv_r6d: # (B, L, 6) -> 142:148  gv
                local_transl_vel,  # (B, L, 3) -> 148:151, smpl-coord
            }
        """
        B, L = inputs["smpl_params_c"]["body_pose"].shape[:2]
        # cam
        smpl_params_c = inputs["smpl_params_c"]
        body_pose = smpl_params_c["body_pose"].reshape(B, L, 21, 3)
        body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).flatten(-2)
        betas = smpl_params_c["betas"]
        global_orient_R = axis_angle_to_matrix(smpl_params_c["global_orient"])
        global_orient_r6d = matrix_to_rotation_6d(global_orient_R)

        # global
        R_c2gv = inputs["R_c2gv"]  # (B, L, 3, 3)
        global_orient_gv_r6d = matrix_to_rotation_6d(R_c2gv @ global_orient_R)

        # local_transl_vel
        smpl_params_w = inputs["smpl_params_w"]
        local_transl_vel = get_local_transl_vel(smpl_params_w["transl"], smpl_params_w["global_orient"])
        if False:  # debug
            transl_recover = rollout_local_transl_vel(
                local_transl_vel, smpl_params_w["global_orient"], smpl_params_w["transl"][:, [0]]
            )
            print((transl_recover - smpl_params_w["transl"]).abs().max())

        # returns
        x = torch.cat([body_pose_r6d, betas, global_orient_r6d, global_orient_gv_r6d, local_transl_vel], dim=-1)
        x_norm = (x - self.mean) / self.std
        return x_norm

    def encode_translw(self, inputs):
        """
        definition: {
                body_pose_r6d,  # (B, L, (J-1)*6) -> 0:126
                betas, # (B, L, 10) -> 126:136
                global_orient_r6d,  # (B, L, 6) -> 136:142  incam
                global_orient_gv_r6d: # (B, L, 6) -> 142:148  gv
                local_transl_vel,  # (B, L, 3) -> 148:151, smpl-coord
            }
        """
        # local_transl_vel
        smpl_params_w = inputs["smpl_params_w"]
        local_transl_vel = get_local_transl_vel(smpl_params_w["transl"], smpl_params_w["global_orient"])

        # returns
        x = local_transl_vel
        x_norm = (x - self.mean[-3:]) / self.std[-3:]
        return x_norm

    def decode_translw(self, x_norm):
        return x_norm * self.std[-3:] + self.mean[-3:]

    def decode(self, x_norm):
        """x_norm: (B, L, C)"""
        B, L, C = x_norm.shape
        x = (x_norm * self.std) + self.mean

        body_pose_r6d = x[:, :, :126]
        betas = x[:, :, 126:136]
        global_orient_r6d = x[:, :, 136:142]
        global_orient_gv_r6d = x[:, :, 142:148]
        local_transl_vel = x[:, :, 148:151]

        body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(body_pose_r6d.reshape(B, L, -1, 6)))
        body_pose = body_pose.flatten(-2)
        global_orient_c = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_r6d))
        global_orient_gv = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_gv_r6d))

        output = {
            "body_pose": body_pose,
            "betas": betas,
            "global_orient": global_orient_c,
            "global_orient_gv": global_orient_gv,
            "local_transl_vel": local_transl_vel,
        }

        return output


group_name = "endecoder/gvhmr"
cfg_base = builds(EnDecoder, populate_full_signature=True)
MainStore.store(name="v1_no_stdmean", node=cfg_base, group=group_name)
MainStore.store(name="v1", node=cfg_base(stats_name="MM_V1"), group=group_name)
MainStore.store(
    name="v1_amass_local_bedlam_cam",
    node=cfg_base(stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM"),
    group=group_name,
)

MainStore.store(name="v2", node=cfg_base(stats_name="MM_V2"), group=group_name)
MainStore.store(name="v2_1", node=cfg_base(stats_name="MM_V2_1"), group=group_name)
