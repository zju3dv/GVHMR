import torch
import numpy as np
from pathlib import Path
from hmr4d.configs import MainStore, builds

from hmr4d.utils.pylogger import Log
from hmr4d.dataset.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from hmr4d.utils import matrix
from hmr4d.utils.smplx_utils import make_smplx
from tqdm import tqdm

from hmr4d.utils.geo_transform import compute_cam_angvel, apply_T_on_points
from hmr4d.utils.geo.hmr_global import get_tgtcoord_rootparam, get_T_w2c_from_wcparams, get_c_rootparam, get_R_c2gv

from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.vis.renderer import Renderer
import imageio
from hmr4d.utils.video_io_utils import read_video_np
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict


class H36mSmplDataset(ImgfeatMotionDatasetBase):
    def __init__(
        self,
        root="inputs/H36M/hmr4d_support",
        original_coord="az",
        motion_frames=120,  # H36M's videos are 25fps and very long
        lazy_load=False,
    ):
        # Path
        self.root = Path(root)

        # Coord
        self.original_coord = original_coord

        # Setting
        self.motion_frames = motion_frames
        self.lazy_load = lazy_load

        super().__init__()

    def _load_dataset(self):
        # smplpose
        tic = Log.time()
        fn = self.root / "smplxpose_v1.pt"
        self.smpl_model = make_smplx("supermotion")
        Log.info(f"[H36M] Loading from {fn} ...")
        self.motion_files = torch.load(fn)
        # Dict of {
        #          "smpl_params_glob": {'body_pose', 'global_orient', 'transl', 'betas'}, FxC
        #          "cam_Rt": tensor(F, 3),
        #          "cam_K": tensor(1, 10),
        #         }
        self.seqs = list(self.motion_files.keys())
        Log.info(f"[H36M] {len(self.seqs)} sequences. Elapsed: {Log.time() - tic:.2f}s")

        # img(as feature)
        # vid -> (features, vid, meta {bbx_xys, K_fullimg})
        if not self.lazy_load:
            tic = Log.time()
            fn = self.root / "vitfeat_h36m.pt"
            Log.info(f"[H36M] Fully Loading to RAM ViT-Feat: {fn}")
            self.f_img_dicts = torch.load(fn)
            Log.info(f"[H36M] Finished. Elapsed: {Log.time() - tic:.2f}s")
        else:
            raise NotImplementedError  # "Check BEDLAM-SMPL for lazy_load"

    def _get_idx2meta(self):
        # We expect to see the entire sequence during one epoch,
        # so each sequence will be sampled max(SeqLength // MotionFrames, 1) times
        seq_lengths = []
        self.idx2meta = []
        for vid in self.f_img_dicts:
            seq_length = self.f_img_dicts[vid]["bbx_xys"].shape[0]
            num_samples = max(seq_length // self.motion_frames, 1)
            seq_lengths.append(seq_length)
            self.idx2meta.extend([vid] * num_samples)
        hours = sum(seq_lengths) / 25 / 3600
        Log.info(f"[H36M] has {hours:.1f} hours motion -> Resampled to {len(self.idx2meta)} samples.")

    def _load_data(self, idx):
        sampled_motion = {}
        vid = self.idx2meta[idx]
        motion = self.motion_files[vid]
        seq_length = self.f_img_dicts[vid]["bbx_xys"].shape[0]  # this is a better choice
        sampled_motion["vid"] = vid

        # Random select a subset
        target_length = self.motion_frames
        if target_length > seq_length:  # this should not happen
            start = 0
            length = seq_length
            Log.info(f"[H36M] ({idx}) target length < sequence length: {target_length} <= {seq_length}")
        else:
            start = np.random.randint(0, seq_length - target_length)
            length = target_length
        end = start + length
        sampled_motion["length"] = length
        sampled_motion["start_end"] = (start, end)

        # Select motion subset
        # body_pose, global_orient, transl, betas
        sampled_motion["smpl_params_global"] = {k: v[start:end] for k, v in motion["smpl_params_glob"].items()}

        # Image as feature
        f_img_dict = self.f_img_dicts[vid]
        sampled_motion["f_imgseq"] = f_img_dict["features"][start:end].float()  # (L, 1024)
        sampled_motion["bbx_xys"] = f_img_dict["bbx_xys"][start:end]
        sampled_motion["K_fullimg"] = f_img_dict["K_fullimg"]
        # sampled_motion["kp2d"] = self.vitpose[vid][start:end].float()  # (L, 17, 3)
        sampled_motion["kp2d"] = torch.zeros((end - start), 17, 3)  # (L, 17, 3)

        # Camera
        sampled_motion["T_w2c"] = motion["cam_Rt"]  # (4, 4)

        return sampled_motion

    def _process_data(self, data, idx):
        length = data["length"]

        # SMPL params in world
        smpl_params_w = data["smpl_params_global"].copy()  # in az

        # SMPL params in cam
        T_w2c = data["T_w2c"]  # (4, 4)
        offset = self.smpl_model.get_skeleton(smpl_params_w["betas"][0])[0]  # (3)
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
        gravity_vec = torch.tensor([0, 0, -1]).float()  # (3), H36M is az
        T_w2c = T_w2c.repeat(length, 1, 1)  # (F, 4, 4)
        R_c2gv = get_R_c2gv(T_w2c[..., :3, :3], axis_gravity_in_w=gravity_vec)  # (F, 3, 3)

        # Image
        bbx_xys = data["bbx_xys"]  # (F, 3)
        K_fullimg = data["K_fullimg"].repeat(length, 1, 1)  # (F, 3, 3)
        f_imgseq = data["f_imgseq"]  # (F, 1024)
        cam_angvel = compute_cam_angvel(T_w2c[:, :3, :3])  # (F, 6)  slightly different from WHAM

        # Returns: do not forget to make it batchable! (last lines)
        max_len = self.motion_frames
        return_data = {
            "meta": {"data_name": "h36m", "idx": idx, "vid": data["vid"]},
            "length": length,
            "smpl_params_c": smpl_params_c,
            "smpl_params_w": smpl_params_w,
            "R_c2gv": R_c2gv,  # (F, 3, 3)
            "gravity_vec": gravity_vec,  # (3)
            "bbx_xys": bbx_xys,  # (F, 3)
            "K_fullimg": K_fullimg,  # (F, 3, 3)
            "f_imgseq": f_imgseq,  # (F, D)
            "kp2d": data["kp2d"],  # (F, 17, 3)
            "cam_angvel": cam_angvel,  # (F, 6)
            "mask": {
                "valid": get_valid_mask(max_len, length),
                "vitpose": False,
                "bbx_xys": True,
                "f_imgseq": True,
                "spv_incam_only": False,
            },
        }

        if False:  # Render to image to check
            smplx_out = self.smplx(**smpl_params_c)
            # ----- Overlay ----- #
            mid = return_data["meta"]["mid"]
            video_path = self.root / f"videos/{mid}.mp4"
            images = read_video_np(video_path, data["start_end"][0], data["start_end"][1])
            render_dict = {
                "K": K_fullimg[:1],  # only support batch size 1
                "faces": self.smplx.faces,
                "verts": smplx_out.vertices,
                "background": images,
            }
            img_overlay = simple_render_mesh_background(render_dict)
            save_video(img_overlay, f"tmp.mp4")

        # Batchable
        return_data["smpl_params_c"] = repeat_to_max_len_dict(return_data["smpl_params_c"], max_len)
        return_data["smpl_params_w"] = repeat_to_max_len_dict(return_data["smpl_params_w"], max_len)
        return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
        return_data["bbx_xys"] = repeat_to_max_len(return_data["bbx_xys"], max_len)
        return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
        return_data["f_imgseq"] = repeat_to_max_len(return_data["f_imgseq"], max_len)
        return_data["kp2d"] = repeat_to_max_len(return_data["kp2d"], max_len)
        return_data["cam_angvel"] = repeat_to_max_len(return_data["cam_angvel"], max_len)

        return return_data


group_name = "train_datasets/imgfeat_h36m"
node_v1 = builds(H36mSmplDataset)
MainStore.store(name="v1", node=node_v1, group=group_name)
