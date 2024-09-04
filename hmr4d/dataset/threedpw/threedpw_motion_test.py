import torch
from torch.utils import data
from pathlib import Path

from hmr4d.utils.pylogger import Log
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.geo.hmr_cam import estimate_K, resize_K
from hmr4d.utils.geo.flip_utils import flip_kp2d_coco17

from hmr4d.configs import MainStore, builds

VID_HARD = []
# VID_HARD = ["downtown_bar_00_1"]


class ThreedpwSmplFullSeqDataset(data.Dataset):
    def __init__(self, flip_test=False, skip_invalid=False):
        super().__init__()
        self.dataset_name = "3DPW"
        self.skip_invalid = skip_invalid
        Log.info(f"[{self.dataset_name}] Full sequence")

        # Load evaluation protocol from WHAM labels
        self.threedpw_dir = Path("inputs/3DPW/hmr4d_support")
        # ['vname', 'K_fullimg', 'T_w2c', 'smpl_params', 'gender', 'mask_raw', 'mask_wham', 'img_wh']
        self.labels = torch.load(self.threedpw_dir / "test_3dpw_gt_labels.pt")
        self.vid2bbx = torch.load(self.threedpw_dir / "preproc_test_bbx.pt")
        self.vid2kp2d = torch.load(self.threedpw_dir / "preproc_test_kp2d_v0.pt")

        # Setup dataset index
        self.idx2meta = list(self.labels)
        if len(VID_HARD) > 0:  # Pick subsets for fast testing
            self.idx2meta = VID_HARD
        Log.info(f"[{self.dataset_name}] {len(self.idx2meta)} sequences.")

        # If flip_test is enabled, we will return extra data for flipped test
        self.flip_test = flip_test
        if self.flip_test:
            Log.info(f"[{self.dataset_name}] Flip test enabled")

    def __len__(self):
        return len(self.idx2meta)

    def _load_data(self, idx):
        data = {}
        vid = self.idx2meta[idx]
        meta = {"dataset_id": self.dataset_name, "vid": vid}
        data.update({"meta": meta})

        # Add useful data
        label = self.labels[vid]
        mask = label["mask_wham"]
        width_height = label["img_wh"]
        data.update(
            {
                "length": len(mask),  # F
                "smpl_params": label["smpl_params"],  # world
                "gender": label["gender"],  # str
                "T_w2c": label["T_w2c"],  # (F, 4, 4)
                "mask": mask,  # (F)
            }
        )
        K_fullimg = label["K_fullimg"]  # (3, 3)
        if False:
            K_fullimg = estimate_K(*width_height)
        data["K_fullimg"] = K_fullimg

        # Preprocessed:  bbx, kp2d, image as feature
        bbx_xys = self.vid2bbx[vid]["bbx_xys"]  # (F, 3)
        kp2d = self.vid2kp2d[vid]  # (F, 17, 3)
        cam_angvel = compute_cam_angvel(data["T_w2c"][:, :3, :3])  # (L, 6)
        data.update({"bbx_xys": bbx_xys, "kp2d": kp2d, "cam_angvel": cam_angvel})

        imgfeat_dir = self.threedpw_dir / "imgfeats/3dpw_test"
        f_img_dict = torch.load(imgfeat_dir / f"{vid}.pt")
        f_imgseq = f_img_dict["features"].float()
        data["f_imgseq"] = f_imgseq  # (F, 1024)

        # to render a video
        vname = label["vname"]
        video_path = self.threedpw_dir / f"videos/{vname}.mp4"
        frame_id = torch.where(mask)[0].long()
        ds = 0.5
        K_render = resize_K(K_fullimg, ds)
        bbx_xys_render = bbx_xys * ds
        kp2d_render = kp2d.clone()
        kp2d_render[..., :2] *= ds
        data["meta_render"] = {
            "name": vid,
            "video_path": str(video_path),
            "ds": ds,
            "frame_id": frame_id,
            "K": K_render,
            "bbx_xys": bbx_xys_render,
            "kp2d": kp2d_render,
        }

        if self.flip_test:
            imgfeat_dir = self.threedpw_dir / "imgfeats/3dpw_test_flip"
            f_img_dict = torch.load(imgfeat_dir / f"{vid}.pt")
            flipped_bbx_xys = f_img_dict["bbx_xys"].float()  # (L, 3)
            flipped_features = f_img_dict["features"].float()  # (L, 1024)
            flipped_kp2d = flip_kp2d_coco17(kp2d, width_height[0])  # (L, 17, 3)

            R_flip_x = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()
            flipped_R_w2c = R_flip_x @ data["T_w2c"][:, :3, :3].clone()

            data_flip = {
                "bbx_xys": flipped_bbx_xys,
                "f_imgseq": flipped_features,
                "kp2d": flipped_kp2d,
                "cam_angvel": compute_cam_angvel(flipped_R_w2c),
            }
            data["flip_test"] = data_flip
        return data

    def _process_data(self, data):
        length = data["length"]
        data["K_fullimg"] = data["K_fullimg"][None].repeat(length, 1, 1)

        if self.skip_invalid:  # Drop all invalid frames
            mask = data["mask"].clone()
            data["length"] = sum(mask)
            data["smpl_params"] = {k: v[mask].clone() for k, v in data["smpl_params"].items()}
            data["T_w2c"] = data["T_w2c"][mask].clone()
            data["mask"] = data["mask"][mask].clone()
            data["K_fullimg"] = data["K_fullimg"][mask].clone()
            data["bbx_xys"] = data["bbx_xys"][mask].clone()
            data["kp2d"] = data["kp2d"][mask].clone()
            data["cam_angvel"] = data["cam_angvel"][mask].clone()
            data["f_imgseq"] = data["f_imgseq"][mask].clone()
            data["flip_test"] = {k: v[mask].clone() for k, v in data["flip_test"].items()}

        return data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data)
        return data


# 3DPW
MainStore.store(
    name="fliptest",
    node=builds(ThreedpwSmplFullSeqDataset, flip_test=True),
    group="test_datasets/3dpw",
)
MainStore.store(
    name="v1",
    node=builds(ThreedpwSmplFullSeqDataset, flip_test=False),
    group="test_datasets/3dpw",
)
