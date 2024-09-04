import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle
import torch

RESOURCE_FOLDER = Path(__file__).resolve().parent / "resource"

camera_idx_to_name = {0: "54138969", 1: "55011271", 2: "58860488", 3: "60457274"}


def get_vid(pkl_path, cam_id):
    """.../S6/Posing 1.pkl, 54138969 -> S6@Posing_1@54138969"""
    sub_id, fn = pkl_path.split("/")[-2:]
    vid = f"{sub_id}@{fn.split('.')[0].replace(' ', '_')}@{cam_id}"
    return vid


def get_raw_pkl_paths(h36m_raw_root):
    smpl_param_dir = h36m_raw_root / "neutrSMPL_H3.6"
    pkl_paths = []
    for train_sub in ["S1", "S5", "S6", "S7", "S8"]:
        for pth in (smpl_param_dir / train_sub).glob("*.pkl"):
            if "aligned" not in str(pth):  # Use world sequence only
                pkl_paths.append(str(pth))

    return pkl_paths


def get_cam_KRts():
    """
    Returns:
        Ks (torch.Tensor): {cam_id: 3x3}
        Rts (torch.Tensor): {subj_id: {cam_id: 4x4}}
    """
    # this file is copied from https://github.com/karfly/human36m-camera-parameters
    cameras_path = RESOURCE_FOLDER / "camera-parameters.json"
    with open(cameras_path, "r") as f:
        cameras = json.load(f)

    # 4 camera ids: '54138969', '55011271', '58860488', '60457274'
    Ks = {}
    for cam in cameras["intrinsics"]:
        Ks[cam] = torch.tensor(cameras["intrinsics"][cam]["calibration_matrix"]).float()

    # extrinsics
    extrinsics = cameras["extrinsics"]
    Rts = defaultdict(dict)
    for subj in extrinsics:
        for cam in extrinsics[subj]:
            Rt = torch.eye(4)
            Rt[:3, :3] = torch.tensor(extrinsics[subj][cam]["R"])
            Rt[:3, [3]] = torch.tensor(extrinsics[subj][cam]["t"]) / 1000
            Rts[subj][cam] = Rt.float()

    return Ks, Rts


def parse_raw_pkl(pkl_path, to_50hz=True):
    """
    raw_pkl @ 200Hz, where video @ 50Hz.
    the frames should be divided by 4, and mannually align with the video.
    """
    with open(str(pkl_path), "rb") as f:
        data = pickle.load(f, encoding="bytes")
    poses = torch.from_numpy(data[b"poses"]).float()
    betas = torch.from_numpy(data[b"betas"]).float()
    trans = torch.from_numpy(data[b"trans"]).float()
    assert poses.shape[0] == trans.shape[0]
    if to_50hz:
        poses = poses[::4]
        trans = trans[::4]

    seq_length = poses.shape[0]  # 50FPS
    smpl_params = {
        "body_pose": poses[:, 3:],
        "betas": betas[None].expand(seq_length, -1),
        "global_orient": poses[:, :3],
        "transl": trans,
    }
    return smpl_params
