import torch
from torch.utils import data
from pathlib import Path
import numpy as np

from hmr4d.utils.pylogger import Log
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.geo.hmr_cam import estimate_K, resize_K
from hmr4d.utils.geo.flip_utils import flip_kp2d_coco17
from hmr4d.dataset.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.video_io_utils import get_video_lwh, read_video_np, save_video
from hmr4d.utils.vis.renderer_utils import simple_render_mesh_background

from hmr4d.configs import MainStore, builds


class ThreedpwSmplDataset(ImgfeatMotionDatasetBase):
    def __init__(self):
        # Path
        self.hmr4d_support_dir = Path("inputs/3DPW/hmr4d_support")
        self.dataset_name = "3DPW"

        # Setting
        self.min_motion_frames = 60
        self.max_motion_frames = 120
        super().__init__()

    def _load_dataset(self):
        self.train_labels = torch.load(self.hmr4d_support_dir / "train_3dpw_gt_labels.pt")
        self.refit_smplx = torch.load(self.hmr4d_support_dir / "train_refit_smplx.pt")
        if True:  # Remove clips that have obvious error
            update_list = {
                "courtyard_basketball_00_1": [(0, 300), (340, 468)],
                "courtyard_laceShoe_00_0": [(0, 620), (780, 931)],
                "courtyard_rangeOfMotions_00_1": [(0, 370), (410, 601)],
                "courtyard_shakeHands_00_1": [(0, 100), (120, 391)],
            }
            for k, v in update_list.items():
                self.refit_smplx[k]["valid_range_list"] = v

        self.f_img_folder = self.hmr4d_support_dir / "imgfeats/3dpw_train_smplx_refit"
        Log.info(f"[{self.dataset_name}] Train")

    def _get_idx2meta(self):
        # We expect to see the entire sequence during one epoch,
        # so each sequence will be sampled max(SeqLength // MotionFrames, 1) times
        seq_lengths = []
        self.idx2meta = []
        for vid in self.refit_smplx:
            valid_range_list = self.refit_smplx[vid]["valid_range_list"]
            for start, end in valid_range_list:
                seq_length = end - start
                num_samples = max(seq_length // self.max_motion_frames, 1)
                seq_lengths.append(seq_length)
                self.idx2meta.extend([(vid, start, end)] * num_samples)
        minutes = sum(seq_lengths) / 25 / 60
        Log.info(
            f"[{self.dataset_name}] has {minutes:.1f} minutes motion -> Resampled to {len(self.idx2meta)} samples."
        )

    def _load_data(self, idx):
        data = {}
        vid, range1, range2 = self.idx2meta[idx]

        # Random select a subset
        mlength = range2 - range1
        min_motion_len = self.min_motion_frames
        max_motion_len = self.max_motion_frames

        if mlength < min_motion_len:  # this may happen, the minimal mlength is around 30
            start = range1
            length = mlength
        else:
            effect_max_motion_len = min(max_motion_len, mlength)
            length = np.random.randint(min_motion_len, effect_max_motion_len + 1)  # [low, high)
            start = np.random.randint(range1, range2 - length + 1)
        end = start + length
        data["length"] = length
        data["meta"] = {"data_name": self.dataset_name, "idx": idx, "vid": vid, "start_end": (start, end)}

        # Select motion subset
        data["smplx_params_incam"] = {k: v[start:end] for k, v in self.refit_smplx[vid]["smplx_params_incam"].items()}
        data["K_fullimg"] = self.train_labels[vid]["K_fullimg"]
        data["T_w2c"] = self.train_labels[vid]["T_w2c"][start:end]

        # Img (as feature):
        f_img_dict = torch.load(self.f_img_folder / f"{vid}.pt")
        data["bbx_xys"] = f_img_dict["bbx_xys"][start:end]  # (F, 3)
        data["f_imgseq"] = f_img_dict["features"][start:end].float()  # (F, 3)
        data["img_wh"] = f_img_dict["img_wh"]  # (2)
        data["kp2d"] = torch.zeros((end - start), 17, 3)  # (L, 17, 3)  # do not provide kp2d

        return data

    def _process_data(self, data, idx):
        length = data["length"]

        smpl_params_c = data["smplx_params_incam"]
        smpl_params_w_zero = {k: torch.zeros_like(v) for k, v in smpl_params_c.items()}
        K_fullimg = data["K_fullimg"][None].repeat(length, 1, 1)
        cam_angvel = compute_cam_angvel(data["T_w2c"][:, :3, :3])

        max_len = self.max_motion_frames
        return_data = {
            "meta": data["meta"],
            "length": length,
            "smpl_params_c": smpl_params_c,
            "smpl_params_w": smpl_params_w_zero,
            "R_c2gv": torch.zeros(length, 3, 3),  # (F, 3, 3)
            "gravity_vec": torch.zeros(3),  # (3)
            "bbx_xys": data["bbx_xys"],  # (F, 3)
            "K_fullimg": K_fullimg,  # (F, 3, 3)
            "f_imgseq": data["f_imgseq"],  # (F, D)
            "kp2d": data["kp2d"],  # (F, 17, 3)
            "cam_angvel": cam_angvel,  # (F, 6)
            "mask": {
                "valid": get_valid_mask(max_len, length),
                "vitpose": False,
                "bbx_xys": True,
                "f_imgseq": True,
                "spv_incam_only": True,
            },
        }

        if False:  # Debug, render incam
            start, end = data["meta"]["start_end"]
            vid = data["meta"]["vid"]

            ds = 0.5
            faces = smplx.faces
            smplx = make_smplx("supermotion")
            smplx_c_verts = smplx(**return_data["smpl_params_c"]).vertices
            K_render = resize_K(K_fullimg, ds)

            video_path = self.hmr4d_support_dir / f"videos/{vid[:-2]}.mp4"
            images = read_video_np(video_path, scale=ds, start_frame=start, end_frame=end)

            render_dict = {
                "K": K_render[:1],  # only support batch size 1
                "faces": faces,
                "verts": smplx_c_verts,
                "background": images,
            }
            img_overlay = simple_render_mesh_background(render_dict, VI=10)
            save_video(img_overlay, f"tmp.mp4", crf=28)

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


# 3DPW
MainStore.store(name="v1", node=builds(ThreedpwSmplDataset), group="train_datasets/imgfeat_3dpw")
