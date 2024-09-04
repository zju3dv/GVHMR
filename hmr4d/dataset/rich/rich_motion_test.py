from pathlib import Path
import numpy as np
import torch
from torch.utils import data
from hmr4d.utils.pylogger import Log

from .rich_utils import (
    get_cam2params,
    get_w2az_sahmr,
    parse_seqname_info,
    get_cam_key_wham_vid,
)
from hmr4d.utils.geo_transform import apply_T_on_points, transform_mat, compute_cam_angvel
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.smplx_utils import make_smplx
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from hmr4d.utils.geo.hmr_cam import resize_K


from hmr4d.configs import MainStore, builds


VID_PRESETS = {
    "easytohard": [
        "test/Gym_013_burpee4/cam_06",
        "test/Gym_011_pushup1/cam_02",
        "test/LectureHall_019_wipingchairs1/cam_03",
        "test/ParkingLot2_009_overfence1/cam_04",
        "test/LectureHall_021_sidebalancerun1/cam_00",
        "test/Gym_010_dips2/cam_05",
    ],
}


class RichSmplFullSeqDataset(data.Dataset):
    def __init__(self, vid_presets=None):
        """
        Args:
            vid_presets is a key in VID_PRESETS
        """
        super().__init__()
        self.dataset_name = "RICH"
        self.dataset_id = "RICH"
        Log.info(f"[{self.dataset_name}] Full sequence, Test")
        tic = Log.time()

        # Load evaluation protocol from WHAM labels
        self.rich_dir = Path("inputs/RICH/hmr4d_support")
        self.labels = torch.load(self.rich_dir / "rich_test_labels.pt")
        self.preproc_data = torch.load(self.rich_dir / "rich_test_preproc.pt")
        vids = select_subset(self.labels, vid_presets)

        # Setup dataset index
        self.idx2meta = []
        for vid in vids:
            seq_length = len(self.labels[vid]["frame_id"])
            self.idx2meta.append((vid, 0, seq_length))  # start=0, end=seq_length
        # print(sum([end - start for _, _, start, end in self.idx2meta]))

        # Prepare ground truth motion in ay-coordinate
        self.w2az = get_w2az_sahmr()  # scan_name -> T_w2az, w-coordinate refers to cam-1-coordinate
        self.cam2params = get_cam2params()  # cam_key -> (T_w2c, K)
        seqname_info = parse_seqname_info(skip_multi_persons=True)  # {k: (scan_name, subject_id, gender, cam_ids)}
        self.seqname_to_scanname = {k: v[0] for k, v in seqname_info.items()}

        Log.info(f"[RICH] {len(self.idx2meta)} sequences. Elapsed: {Log.time() - tic:.2f}s")

    def __len__(self):
        return len(self.idx2meta)

    def _load_data(self, idx):
        data = {}

        # [start, end), when loading data from labels
        vid, start, end = self.idx2meta[idx]
        label = self.labels[vid]
        preproc_data = self.preproc_data[vid]

        length = end - start
        meta = {"dataset_id": "RICH", "vid": vid, "vid-start-end": (start, end)}
        data.update({"meta": meta, "length": length})

        # SMPLX
        data.update({"gt_smpl_params": label["gt_smplx_params"], "gender": label["gender"]})

        # camera
        cam_key = get_cam_key_wham_vid(vid)
        scan_name = self.seqname_to_scanname[vid.split("/")[1]]
        T_w2c, K = self.cam2params[cam_key]  # (4, 4)  (3, 3)
        T_w2az = self.w2az[scan_name]
        data.update({"T_w2c": T_w2c, "T_w2az": T_w2az, "K": K})

        # image features
        data.update(
            {
                "f_imgseq": preproc_data["f_imgseq"],
                "bbx_xys": preproc_data["bbx_xys"],
                "img_wh": preproc_data["img_wh"],
                "kp2d": preproc_data["kp2d"],
            }
        )

        # to render a video
        video_path = self.rich_dir / "video" / vid / "video.mp4"
        frame_id = label["frame_id"]  # (F,)
        width, height = data["img_wh"] / 4  #  Video saved has been downsampled 1/4
        K_render = resize_K(K, 0.25)
        bbx_xys_render = data["bbx_xys"] / 4
        data["meta_render"] = {
            "name": vid.replace("/", "@"),
            "video_path": str(video_path),
            "frame_id": frame_id,
            "width_height": (width, height),
            "K": K_render,
            "bbx_xys": bbx_xys_render,
        }

        return data

    def _process_data(self, data):
        # T_w2az is pre-computed by using floor clue. az2zy uses a rotation along x-axis.
        R_az2ay = axis_angle_to_matrix(torch.tensor([1.0, 0.0, 0.0]) * -torch.pi / 2)  # (3, 3)
        T_w2ay = transform_mat(R_az2ay, R_az2ay.new([0, 0, 0])) @ data["T_w2az"]  # (4, 4)

        if False:  #  Visualize groundtruth and observation
            self.rich_smplx = {
                "male": make_smplx("rich-smplx", gender="male"),
                "female": make_smplx("rich-smplx", gender="female"),
            }
            wis3d = make_wis3d(name="debug-rich-smpl_dataset")
            rich_smplx = make_smplx("rich-smplx", gender=data["gender"])
            smplx_out = rich_smplx(**data["gt_smpl_params"])
            smplx_verts_ay = apply_T_on_points(smplx_out.vertices, T_w2ay)
            for i in range(400):
                wis3d.set_scene_id(i)
                wis3d.add_mesh(smplx_out.vertices[i], rich_smplx.bm.faces, name=f"gt-smplx")
                wis3d.add_mesh(smplx_verts_ay[i], rich_smplx.bm.faces, name=f"gt-smplx-ay")

        # process img feature with xys
        length = data["length"]
        f_imgseq = data["f_imgseq"]  # (F, 1024)
        R_w2c = data["T_w2c"][:3, :3].repeat(length, 1, 1)  # (L, 4, 4)
        cam_angvel = compute_cam_angvel(R_w2c)  # (L, 6)

        # Return
        data = {
            # --- not batched
            "task": "CAP-Seq",
            "meta": data["meta"],
            "meta_render": data["meta_render"],
            # --- we test on single sequence, so set kv manually
            "length": length,
            "f_imgseq": f_imgseq,
            "cam_angvel": cam_angvel,
            "bbx_xys": data["bbx_xys"],  # (F, 3)
            "K_fullimg": data["K"][None].expand(length, -1, -1),  # (F, 3, 3)
            "kp2d": data["kp2d"],  # (F, 17, 3)
            # --- dataset specific
            "model": "smplx",
            "gender": data["gender"],
            "gt_smpl_params": data["gt_smpl_params"],
            "T_w2ay": T_w2ay,  # (4, 4)
            "T_w2c": data["T_w2c"],  # (4, 4)
        }
        return data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data)
        return data


def select_subset(labels, vid_presets):
    vids = list(labels.keys())
    if vid_presets != None:  # Use a subset of the videos
        vids = VID_PRESETS[vid_presets]
    return vids


#
group_name = "test_datasets/rich"
base_node = builds(RichSmplFullSeqDataset, vid_presets=None, populate_full_signature=True)
MainStore.store(name="all", node=base_node, group=group_name)
MainStore.store(name="easy_to_hard", node=base_node(vid_presets="easytohard"), group=group_name)
MainStore.store(name="postproc", node=base_node(vid_presets="postproc"), group=group_name)
