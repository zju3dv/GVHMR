import torch
from torch.utils.data import Dataset
from pathlib import Path

from .utils import *
from .cam_traj_utils import CameraAugmentorV11
from hmr4d.utils.geo.hmr_cam import create_camera_sensor
from hmr4d.utils.geo.hmr_global import get_c_rootparam, get_R_c2gv
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from hmr4d.utils.geo_transform import compute_cam_angvel, apply_T_on_points, project_p2d, cvt_p2d_from_i_to_c

from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines, convert_motion_as_line_mesh
from hmr4d.utils.smplx_utils import make_smplx


class BaseDataset(Dataset):
    def __init__(self, cam_augmentation, limit_size=None):
        super().__init__()
        self.cam_augmentation = cam_augmentation
        self.limit_size = limit_size
        self.smplx = make_smplx("supermotion")
        self.smplx_lite = make_smplx("supermotion_smpl24")

        self._load_dataset()
        self._get_idx2meta()

    def _load_dataset(self):
        NotImplementedError("_load_dataset is not implemented")

    def _get_idx2meta(self):
        self.idx2meta = None
        NotImplementedError("_get_idx2meta is not implemented")

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)

    def _load_data(self, idx):
        NotImplementedError("_load_data is not implemented")

    def _process_data(self, data, idx):
        """
        Args:
            data: dict {
                "body_pose": (F, 63),
                "betas": (F, 10),
                "global_orient": (F, 3),  in the AY coordinates
                "transl": (F, 3),  in the AY coordinates
            }
        """
        data_name = data["data_name"]
        length = data["body_pose"].shape[0]
        # Augmentation: betas, SMPL (gravity-axis)
        body_pose = data["body_pose"]
        betas = augment_betas(data["betas"], std=0.1)
        global_orient_w, transl_w = rotate_around_axis(data["global_orient"], data["transl"], axis="y")
        del data

        # SMPL_params in world
        smpl_params_w = {
            "body_pose": body_pose,  # (F, 63)
            "betas": betas,  # (F, 10)
            "global_orient": global_orient_w,  # (F, 3)
            "transl": transl_w,  # (F, 3)
        }

        # Camera trajectory augmentation
        if self.cam_augmentation == "v11":
            # interleave repeat to original length (faster)
            N = 10
            w_j3d = self.smplx_lite(
                smpl_params_w["body_pose"][::N],
                smpl_params_w["betas"][::N],
                smpl_params_w["global_orient"][::N],
                None,
            )
            w_j3d = w_j3d.repeat_interleave(N, dim=0) + smpl_params_w["transl"][:, None]  # (F, 24, 3)

            if False:
                wis3d = make_wis3d(name="debug_amass")
                add_motion_as_lines(w_j3d, wis3d, "w_j3d")

            width, height, K_fullimg = create_camera_sensor(1000, 1000, 43.3)  # WHAM
            focal_length = K_fullimg[0, 0]
            wham_cam_augmentor = CameraAugmentorV11()
            T_w2c = wham_cam_augmentor(w_j3d, length)  # (F, 4, 4)

        else:
            raise NotImplementedError

        if False:  # render
            for idx_render in range(10):
                T_w2c = wham_cam_augmentor(smpl_params_w["transl"])

                # targets
                w_j3d = self.smplx(**smpl_params_w).joints[:, :22]
                c_j3d = apply_T_on_points(w_j3d, T_w2c)
                verts, faces, vertex_colors = convert_motion_as_line_mesh(c_j3d)
                vertex_colors = vertex_colors[None] / 255.0
                bg = np.ones((height, width, 3), dtype=np.uint8) * 255

                # render
                renderer = Renderer(width, height, device="cuda", faces=faces, K=K_fullimg)
                vname = f"{idx_render:02d}"
                out_fn = Path(f"outputs/dump_render_wham_cam/{vname}.mp4")
                out_fn.parent.mkdir(exist_ok=True, parents=True)
                writer = imageio.get_writer(out_fn, fps=30, mode="I", format="FFMPEG", macro_block_size=1)
                for i in tqdm(range(len(verts)), desc=f"Rendering {vname}"):
                    # incam
                    # img_overlay_pred = renderer.render_mesh(verts[i].cuda(), bg, [0.8, 0.8, 0.8], VI=1)
                    img_overlay_pred = renderer.render_mesh(verts[i].cuda(), bg, vertex_colors, VI=1)
                    # if batch["meta_render"][0].get("bbx_xys", None) is not None:  # draw bbox lines
                    #     bbx_xys = batch["meta_render"][0]["bbx_xys"][i].cpu().numpy()
                    #     lu_point = (bbx_xys[:2] - bbx_xys[2:] / 2).astype(int)
                    #     rd_point = (bbx_xys[:2] + bbx_xys[2:] / 2).astype(int)
                    #     img_overlay_pred = cv2.rectangle(img_overlay_pred, lu_point, rd_point, (255, 178, 102), 2)

                    # write
                    writer.append_data(img_overlay_pred)
                writer.close()
                pass

        # SMPL params in cam
        offset = self.smplx.get_skeleton(smpl_params_w["betas"][0])[0]  # (3)
        global_orient_c, transl_c = get_c_rootparam(
            smpl_params_w["global_orient"],
            smpl_params_w["transl"],
            T_w2c,
            offset,
        )
        smpl_params_c = {
            "body_pose": smpl_params_w["body_pose"].clone(),  # (F, 63)
            "betas": smpl_params_w["betas"].clone(),  # (F, 10)
            "global_orient": global_orient_c,  # (F, 3)
            "transl": transl_c,  # (F, 3)
        }

        # World params
        gravity_vec = torch.tensor([0, -1, 0], dtype=torch.float32)  # (3), BEDLAM is ay
        R_c2gv = get_R_c2gv(T_w2c[:, :3, :3], gravity_vec)  # (F, 3, 3)

        # Image
        K_fullimg = K_fullimg.repeat(length, 1, 1)  # (F, 3, 3)
        cam_angvel = compute_cam_angvel(T_w2c[:, :3, :3])  # (F, 6)

        # Returns: do not forget to make it batchable! (last lines)
        # NOTE: bbx_xys and f_imgseq will be added later
        max_len = length
        return_data = {
            "meta": {"data_name": data_name, "idx": idx, "T_w2c": T_w2c},
            "length": length,
            "smpl_params_c": smpl_params_c,
            "smpl_params_w": smpl_params_w,
            "R_c2gv": R_c2gv,  # (F, 3, 3)
            "gravity_vec": gravity_vec,  # (3)
            "bbx_xys": torch.zeros((length, 3)),  # (F, 3)  # NOTE: a placeholder
            "K_fullimg": K_fullimg,  # (F, 3, 3)
            "f_imgseq": torch.zeros((length, 1024)),  # (F, D)  # NOTE: a placeholder
            "kp2d": torch.zeros(length, 17, 3),  # (F, 17, 3)
            "cam_angvel": cam_angvel,  # (F, 6)
            "mask": {
                "valid": get_valid_mask(length, length),
                "vitpose": False,
                "bbx_xys": False,
                "f_imgseq": False,
                "spv_incam_only": False,
            },
        }

        # Batchable
        return_data["smpl_params_c"] = repeat_to_max_len_dict(return_data["smpl_params_c"], max_len)
        return_data["smpl_params_w"] = repeat_to_max_len_dict(return_data["smpl_params_w"], max_len)
        return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
        return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
        return_data["cam_angvel"] = repeat_to_max_len(return_data["cam_angvel"], max_len)
        return return_data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data
