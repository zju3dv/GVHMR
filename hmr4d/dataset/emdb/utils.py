import torch
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from hmr4d.utils.geo_transform import convert_lurb_to_bbx_xys
from hmr4d.utils.video_io_utils import get_video_lwh


def name_to_subfolder(name):
    return f"{name[:2]}/{name[3:]}"


def name_to_local_pkl_path(name):
    return f"{name_to_subfolder(name)}/{name}_data.pkl"


def load_raw_pkl(fp):
    annot = pickle.load(open(fp, "rb"))
    annot["subfolder"] = name_to_subfolder(annot["name"])
    return annot


def load_pkl(fp):
    annot = pickle.load(open(fp, "rb"))
    # ['gender', 'name', 'emdb1', 'emdb2', 'n_frames', 'good_frames_mask', 'camera', 'smpl', 'kp2d', 'bboxes', 'subfolder']
    data = {}

    F = annot["n_frames"]
    smpl_params = {
        "body_pose": annot["smpl"]["poses_body"],  # (F, 69)
        "betas": annot["smpl"]["betas"][None].repeat(F, axis=0),  # (F, 10)
        "global_orient": annot["smpl"]["poses_root"],  # (F, 3)
        "transl": annot["smpl"]["trans"],  # (F, 3)
    }
    smpl_params = {k: torch.from_numpy(v).float() for k, v in smpl_params.items()}

    data["name"] = annot["name"]
    data["gender"] = annot["gender"]
    data["smpl_params"] = smpl_params
    data["mask"] = torch.from_numpy(annot["good_frames_mask"]).bool()  # (L,)
    data["K_fullimg"] = torch.from_numpy(annot["camera"]["intrinsics"]).float()  # (3, 3)
    data["T_w2c"] = torch.from_numpy(annot["camera"]["extrinsics"]).float()  # (L, 4, 4)
    bbx_lurb = torch.from_numpy(annot["bboxes"]["bboxes"]).float()
    data["bbx_xys"] = convert_lurb_to_bbx_xys(bbx_lurb)  # (L, 3)

    return data


EMDB1_LIST = [
    "P1/14_outdoor_climb/P1_14_outdoor_climb_data.pkl",
    "P2/23_outdoor_hug_tree/P2_23_outdoor_hug_tree_data.pkl",
    "P3/31_outdoor_workout/P3_31_outdoor_workout_data.pkl",
    "P3/32_outdoor_soccer_warmup_a/P3_32_outdoor_soccer_warmup_a_data.pkl",
    "P3/33_outdoor_soccer_warmup_b/P3_33_outdoor_soccer_warmup_b_data.pkl",
    "P5/42_indoor_dancing/P5_42_indoor_dancing_data.pkl",
    "P5/44_indoor_rom/P5_44_indoor_rom_data.pkl",
    "P6/49_outdoor_big_stairs_down/P6_49_outdoor_big_stairs_down_data.pkl",  # DUPLICATE
    "P6/50_outdoor_workout/P6_50_outdoor_workout_data.pkl",
    "P6/51_outdoor_dancing/P6_51_outdoor_dancing_data.pkl",
    "P7/57_outdoor_rock_chair/P7_57_outdoor_rock_chair_data.pkl",  # DUPLICATE
    "P7/59_outdoor_rom/P7_59_outdoor_rom_data.pkl",
    "P7/60_outdoor_workout/P7_60_outdoor_workout_data.pkl",
    "P8/64_outdoor_skateboard/P8_64_outdoor_skateboard_data.pkl",  # DUPLICATE
    "P8/68_outdoor_handstand/P8_68_outdoor_handstand_data.pkl",
    "P8/69_outdoor_cartwheel/P8_69_outdoor_cartwheel_data.pkl",
    "P9/76_outdoor_sitting/P9_76_outdoor_sitting_data.pkl",
]
EMDB1_NAMES = ["_".join(p.split("/")[:2]) for p in EMDB1_LIST]


EMDB2_LIST = [
    "P0/09_outdoor_walk/P0_09_outdoor_walk_data.pkl",
    "P2/19_indoor_walk_off_mvs/P2_19_indoor_walk_off_mvs_data.pkl",
    "P2/20_outdoor_walk/P2_20_outdoor_walk_data.pkl",
    "P2/24_outdoor_long_walk/P2_24_outdoor_long_walk_data.pkl",
    "P3/27_indoor_walk_off_mvs/P3_27_indoor_walk_off_mvs_data.pkl",
    "P3/28_outdoor_walk_lunges/P3_28_outdoor_walk_lunges_data.pkl",
    "P3/29_outdoor_stairs_up/P3_29_outdoor_stairs_up_data.pkl",
    "P3/30_outdoor_stairs_down/P3_30_outdoor_stairs_down_data.pkl",
    "P4/35_indoor_walk/P4_35_indoor_walk_data.pkl",
    "P4/36_outdoor_long_walk/P4_36_outdoor_long_walk_data.pkl",
    "P4/37_outdoor_run_circle/P4_37_outdoor_run_circle_data.pkl",
    "P5/40_indoor_walk_big_circle/P5_40_indoor_walk_big_circle_data.pkl",
    "P6/48_outdoor_walk_downhill/P6_48_outdoor_walk_downhill_data.pkl",
    "P6/49_outdoor_big_stairs_down/P6_49_outdoor_big_stairs_down_data.pkl",  # DUPLICATE
    "P7/55_outdoor_walk/P7_55_outdoor_walk_data.pkl",
    "P7/56_outdoor_stairs_up_down/P7_56_outdoor_stairs_up_down_data.pkl",
    "P7/57_outdoor_rock_chair/P7_57_outdoor_rock_chair_data.pkl",  # DUPLICATE
    "P7/58_outdoor_parcours/P7_58_outdoor_parcours_data.pkl",
    "P7/61_outdoor_sit_lie_walk/P7_61_outdoor_sit_lie_walk_data.pkl",
    "P8/64_outdoor_skateboard/P8_64_outdoor_skateboard_data.pkl",  # DUPLICATE
    "P8/65_outdoor_walk_straight/P8_65_outdoor_walk_straight_data.pkl",
    "P9/77_outdoor_stairs_up/P9_77_outdoor_stairs_up_data.pkl",
    "P9/78_outdoor_stairs_up_down/P9_78_outdoor_stairs_up_down_data.pkl",
    "P9/79_outdoor_walk_rectangle/P9_79_outdoor_walk_rectangle_data.pkl",
    "P9/80_outdoor_walk_big_circle/P9_80_outdoor_walk_big_circle_data.pkl",
]
EMDB2_NAMES = ["_".join(p.split("/")[:2]) for p in EMDB2_LIST]
EMDB_NAMES = list(sorted(set(EMDB1_NAMES + EMDB2_NAMES)))


def _check_annot(emdb_raw_dir=Path("inputs/EMDB/EMDB")):
    for pkl_local_path in set(EMDB1_LIST + EMDB2_LIST):
        annot = load_raw_pkl(emdb_raw_dir / pkl_local_path)
        if any((annot["bboxes"]["invalid_idxs"] != np.where(~annot["good_frames_mask"])[0])):
            print(annot["name"])


def _check_length(emdb_raw_dir=Path("inputs/EMDB/EMDB"), emdb_hmr4d_support_dir=Path("inputs/EMDB/hmr4d_support")):
    lengths = []
    for local_pkl_path in tqdm(set(EMDB1_LIST + EMDB2_LIST)):
        data = load_pkl(emdb_raw_dir / local_pkl_path)
        video_path = emdb_hmr4d_support_dir / "videos" / f"{data['name']}.mp4"
        length, width, height = get_video_lwh(video_path)
        lengths.append(length)
    print(sorted(lengths))

    video_ram = length[-1] * (width / 4) * (height / 4) * 3 / 1e6
    print(f"Video RAM for {lengths[-1]} x {width} x {height}: {video_ram:.2f} MB")
