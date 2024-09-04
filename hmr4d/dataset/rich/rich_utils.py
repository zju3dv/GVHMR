import torch
import cv2
import numpy as np
from hmr4d.utils.geo_transform import apply_T_on_points, project_p2d
from pathlib import Path
import json
import time

# ----- Meta sample utils ----- #


def sample_idx2meta(idx2meta, sample_interval):
    """
    1. remove frames that < 45
    2. sample frames by sample_interval
    3. sorted
    """
    idx2meta = [
        v
        for k, v in idx2meta.items()
        if int(v["frame_name"]) > 45 and (int(v["frame_name"]) + int(v["cam_id"])) % sample_interval == 0
    ]
    idx2meta = sorted(idx2meta, key=lambda meta: meta["img_key"])
    return idx2meta


def remove_bbx_invisible_frame(idx2meta, img2gtbbx):
    raw_img_lu = np.array([0.0, 0.0])
    raw_img_rb_type1 = np.array([4112.0, 3008.0]) - 1  # horizontal
    raw_img_rb_type2 = np.array([3008.0, 4112.0]) - 1  # vertical

    idx2meta_new = []
    for meta in idx2meta:
        gtbbx_center = np.array([img2gtbbx[meta["img_key"]][[0, 2]].mean(), img2gtbbx[meta["img_key"]][[1, 3]].mean()])
        if (gtbbx_center < raw_img_lu).any():
            continue
        raw_img_rb = raw_img_rb_type1 if meta["cam_key"] not in ["Pavallion_3", "Pavallion_5"] else raw_img_rb_type2
        if (gtbbx_center > raw_img_rb).any():
            continue
        idx2meta_new.append(meta)
    return idx2meta_new


def remove_extra_rules(idx2meta):
    multi_person_seqs = ["LectureHall_009_021_reparingprojector1"]
    idx2meta = [meta for meta in idx2meta if meta["seq_name"] not in multi_person_seqs]
    return idx2meta


# ----- Image utils ----- #


def compute_bbx(dataset, data):
    """
    Use gt_smplh_params to compute bbx (w.r.t. original image resolution)
    Args:
        dataset: rich_pose.RichPose
        data: dict

    # This function need extra scripts to run
    from hmr4d.utils.smplx_utils import make_smplx
    self.smplh_male = make_smplx("rich-smplh", gender="male")
    self.smplh_female = make_smplx("rich-smplh", gender="female")
    self.smplh = {
        "male": self.smplh_male,
        "female": self.smplh_female,
    }
    """
    gender = data["meta"]["gender"]
    smplh_params = {k: v.reshape(1, -1) for k, v in data["gt_smplh_params"].items()}
    smplh_opt = dataset.smplh[gender](**smplh_params)
    verts_3d_w = smplh_opt.vertices
    T_w2c, K = data["T_w2c"], data["K"]
    verts_3d_c = apply_T_on_points(verts_3d_w, T_w2c[None])
    verts_2d = project_p2d(verts_3d_c, K[None])[0]
    min_2d = verts_2d.T.min(-1)[0]
    max_2d = verts_2d.T.max(-1)[0]
    bbx = torch.stack([min_2d, max_2d]).reshape(-1).numpy()
    return bbx


def get_2d(dataset, data):
    gender = data["meta"]["gender"]
    smplh_params = {k: v.reshape(1, -1) for k, v in data["gt_smplh_params"].items()}
    smplh_opt = dataset.smplh[gender](**smplh_params)
    joints_3d_w = smplh_opt.joints
    T_w2c, K = data["T_w2c"], data["K"]
    joints_3d_c = apply_T_on_points(joints_3d_w, T_w2c[None])
    joints_2d = project_p2d(joints_3d_c, K[None])[0]
    conf = torch.ones((73, 1))
    keypoints = torch.cat([joints_2d, conf], dim=1)
    return keypoints


def squared_crop_and_resize(dataset, img, bbx_lurb, dst_size=224, state=None):
    if state is not None:
        np.random.set_state(state)
    center_rand = dataset.BBX_CENTER * (np.random.random(2) * 2 - 1)
    center_x = (bbx_lurb[0] + bbx_lurb[2]) / 2 + center_rand[0]
    center_y = (bbx_lurb[1] + bbx_lurb[3]) / 2 + center_rand[1]
    ori_half_size = max(bbx_lurb[2] - bbx_lurb[0], bbx_lurb[3] - bbx_lurb[1]) / 2
    ori_half_size *= 1 + 0.15 + dataset.BBX_ZOOM * np.random.random()  # zoom

    src = np.array(
        [
            [center_x - ori_half_size, center_y - ori_half_size],
            [center_x + ori_half_size, center_y - ori_half_size],
            [center_x, center_y],
        ],
        dtype=np.float32,
    )
    dst = np.array([[0, 0], [dst_size - 1, 0], [dst_size / 2 - 0.5, dst_size / 2 - 0.5]], dtype=np.float32)

    A = cv2.getAffineTransform(src, dst)
    img_crop = cv2.warpAffine(img, A, (dst_size, dst_size), flags=cv2.INTER_LINEAR)
    bbx_new = np.array(
        [center_x - ori_half_size, center_y - ori_half_size, center_x + ori_half_size, center_y + ori_half_size],
        dtype=bbx_lurb.dtype,
    )
    return img_crop, bbx_new, A


# Augment bbx
def get_augmented_square_bbx(bbx_lurb, per_shift=0.1, per_zoomout=0.2, base_zoomout=0.15, state=None):
    """
    Args:
        per_shift: in percent, maximum random shift
        per_zoomout: in percent, maximum random zoom
    """
    if state is not None:
        np.random.set_state(state)
    maxsize_bbx = max(bbx_lurb[2] - bbx_lurb[0], bbx_lurb[3] - bbx_lurb[1])
    # shift of center
    shift = maxsize_bbx * per_shift * (np.random.random(2) * 2 - 1)
    center_x = (bbx_lurb[0] + bbx_lurb[2]) / 2 + shift[0]
    center_y = (bbx_lurb[1] + bbx_lurb[3]) / 2 + shift[1]
    # zoomout of half-size
    halfsize_bbx = maxsize_bbx / 2
    halfsize_bbx *= 1 + base_zoomout + per_zoomout * np.random.random()

    bbx_lurb = np.array(
        [
            center_x - halfsize_bbx,
            center_y - halfsize_bbx,
            center_x + halfsize_bbx,
            center_y + halfsize_bbx,
        ]
    )
    return bbx_lurb


def get_squared_bbx_region_and_resize(frames, bbx_xys, dst_size=224):
    """
    Args:
        frames: (F, H, W, 3)
        bbx_xys: (F, 3), xys
    """
    frames_np = frames.numpy() if isinstance(frames, torch.Tensor) else frames
    bbx_xys = bbx_xys if isinstance(bbx_xys, torch.Tensor) else torch.tensor(bbx_xys)  # use tensor
    srcs = torch.stack(
        [
            torch.stack([bbx_xys[:, 0] - bbx_xys[:, 2] / 2, bbx_xys[:, 1] - bbx_xys[:, 2] / 2], dim=-1),
            torch.stack([bbx_xys[:, 0] + bbx_xys[:, 2] / 2, bbx_xys[:, 1] - bbx_xys[:, 2] / 2], dim=-1),
            bbx_xys[:, :2],
        ],
        dim=1,
    )  # (F, 3, 2)
    dst = np.array([[0, 0], [dst_size - 1, 0], [dst_size / 2 - 0.5, dst_size / 2 - 0.5]], dtype=np.float32)
    As = np.stack([cv2.getAffineTransform(src, dst) for src in srcs.numpy()])

    img_crops = np.stack(
        [cv2.warpAffine(frames_np[i], As[i], (dst_size, dst_size), flags=cv2.INTER_LINEAR) for i in range(len(As))]
    )
    img_crops = torch.from_numpy(img_crops)
    As = torch.from_numpy(As)
    return img_crops, As


# ----- Camera utils ----- #


def extract_cam_xml(xml_path="", dtype=torch.float32):
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)

    extrinsics_mat = [float(s) for s in tree.find("./CameraMatrix/data").text.split()]
    intrinsics_mat = [float(s) for s in tree.find("./Intrinsics/data").text.split()]
    distortion_vec = [float(s) for s in tree.find("./Distortion/data").text.split()]

    return {
        "ext_mat": torch.tensor(extrinsics_mat).float(),
        "int_mat": torch.tensor(intrinsics_mat).float(),
        "dis_vec": torch.tensor(distortion_vec).float(),
    }


def get_cam2params(scene_info_root=None):
    """
    Args:
        scene_info_root: this could be repalced by path to scan_calibration
    """
    if scene_info_root is not None:
        cam_params = {}
        cam_xml_files = Path(scene_info_root).glob("*/calibration/*.xml")
        for cam_xml_file in cam_xml_files:
            cam_param = extract_cam_xml(cam_xml_file)
            T_w2c = cam_param["ext_mat"].reshape(3, 4)
            T_w2c = torch.cat([T_w2c, torch.tensor([[0, 0, 0, 1.0]])], dim=0)  # (4, 4)
            K = cam_param["int_mat"].reshape(3, 3)
            cap_name = cam_xml_file.parts[-3]
            cam_id = int(cam_xml_file.stem)
            cam_key = f"{cap_name}_{cam_id}"
            cam_params[cam_key] = (T_w2c, K)
    else:
        cam_params = torch.load(Path(__file__).parent / "resource/cam2params.pt")
    return cam_params


# ----- Parse Raw Resource ----- #


def get_w2az_sahmr():
    """
    Returns:
        w2az_sahmr: dict, {scan_name: Tw2az}, Tw2az is a tensor of (4,4)
    """
    fn = Path(__file__).parent / "resource/w2az_sahmr.json"
    with open(fn, "r") as f:
        kvs = json.load(f).items()
    w2az_sahmr = {k: torch.tensor(v) for k, v in kvs}
    return w2az_sahmr


def has_multi_persons(seq_name):
    """
    Args:
        seq_name: e.g. LectureHall_009_021_reparingprojector1
    """
    return len(seq_name.split("_")) != 3


def parse_seqname_info(skip_multi_persons=True):
    """
    This function will skip multi-person sequences.
    Returns:
        sname_to_info: scan_name, subject_id, gender, cam_ids
    """
    fns = [Path(__file__).parent / f"resource/{split}.txt" for split in ["train", "val", "test"]]
    # Train / Val&Test Header:
    # sequence_name	capture_name	scan_name	id	moving_cam	gender	view_id
    # sequence_name	capture_name	scan_name	id	moving_cam	gender	scene	action/scene-interaction	subjects	view_id
    sname_to_info = {}
    for fn in fns:
        with open(fn, "r") as f:
            for line in f.readlines()[1:]:
                raw_values = line.strip().split()
                seq_name = raw_values[0]
                if skip_multi_persons and has_multi_persons(seq_name):
                    continue
                scan_name = f"{raw_values[1]}_{raw_values[2]}"
                subject_id = int(raw_values[3])
                gender = raw_values[5]
                cam_ids = [int(c) for c in raw_values[-1].split(",")]
                sname_to_info[seq_name] = (scan_name, subject_id, gender, cam_ids)
    return sname_to_info


def get_seqnames_of_split(splits=["train"], skip_multi_persons=True):
    if not isinstance(splits, list):
        splits = [splits]
    fns = [Path(__file__).parent / f"resource/{split}.txt" for split in splits]
    seqnames = []
    for fn in fns:
        with open(fn, "r") as f:
            for line in f.readlines()[1:]:
                seq_name = line.strip().split()[0]
                if skip_multi_persons and has_multi_persons(seq_name):
                    continue
                seqnames.append(seq_name)
    return seqnames


def get_seqname_to_imgrange():
    """Each sequence has a different range of image ids."""
    from tqdm import tqdm

    split_seqnames = {split: get_seqnames_of_split(split) for split in ["train", "val", "test"]}
    seqname_to_imgrange = {}
    for split in ["train", "val", "test"]:
        for seqname in tqdm(split_seqnames[split]):
            img_root = Path("inputs/RICH") / "images_ds4" / split  # compressed (not original)
            img_dir = img_root / seqname
            img_names = sorted([n.name for n in img_dir.glob("**/*.jpeg")])
            if len(img_names) == 0:
                img_range = (0, 0)
            else:
                img_range = (int(img_names[0].split("_")[0]), int(img_names[-1].split("_")[0]))
            seqname_to_imgrange[seqname] = img_range
    return seqname_to_imgrange


# ----- Compose keys ----- #


def get_img_key(seq_name, cam_id, f_id):
    assert len(seq_name.split("_")) == 3
    subject_id = int(seq_name.split("_")[1])
    return f"{seq_name}_{int(cam_id)}_{int(f_id):05d}_{subject_id}"


def get_seq_cam_fn(img_root, seq_name, cam_id):
    """
    Args:
        img_root: "inputs/RICH/images_ds4/train"
    """
    img_root = Path(img_root)
    cam_id = int(cam_id)
    return str(img_root / f"{seq_name}/cam_{cam_id:02d}")


def get_img_fn(img_root, seq_name, cam_id, f_id):
    """
    Args:
        img_root: "inputs/RICH/images_ds4/train"
    """
    img_root = Path(img_root)
    cam_id = int(cam_id)
    f_id = int(f_id)
    return str(img_root / f"{seq_name}/cam_{cam_id:02d}" / f"{f_id:05d}_{cam_id:02d}.jpeg")


# ----- WHAM ----- #


def get_cam_key_wham_vid(vid):
    _, sname, cname = vid.split("/")
    scene = sname.split("_")[0]
    cid = int(cname.split("_")[1])
    cam_key = f"{scene}_{cid}"
    return cam_key


def get_K_wham_vid(vid):
    cam_key = get_cam_key_wham_vid(vid)
    cam2params = get_cam2params()
    K = cam2params[cam_key][1]
    return K


class RichVid2Tc2az:
    def __init__(self) -> None:
        self.w2az = get_w2az_sahmr()  # scan_name: tensor 4,4
        seqname_info = parse_seqname_info(skip_multi_persons=True)  # {k: (scan_name, subject_id, gender, cam_ids)}
        self.seqname_to_scanname = {k: v[0] for k, v in seqname_info.items()}
        self.cam2params = get_cam2params()  # cam_key -> (T_w2c, K)

    def __call__(self, vid):
        cam_key = get_cam_key_wham_vid(vid)
        scan_name = self.seqname_to_scanname[vid.split("/")[1]]
        T_w2c, K = self.cam2params[cam_key]  # (4, 4)  (3, 3)
        T_w2az = self.w2az[scan_name]
        T_c2az = T_w2az @ T_w2c.inverse()
        return T_c2az

    def get_T_w2az(self, vid):
        cam_key = get_cam_key_wham_vid(vid)
        scan_name = self.seqname_to_scanname[vid.split("/")[1]]
        T_w2az = self.w2az[scan_name]
        return T_w2az
