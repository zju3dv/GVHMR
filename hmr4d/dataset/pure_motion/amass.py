import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from pathlib import Path
from hmr4d.utils.pylogger import Log
from hmr4d.configs import MainStore, builds

from .base_dataset import BaseDataset
from .utils import *
from hmr4d.utils.geo.hmr_global import get_tgtcoord_rootparam
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines, convert_motion_as_line_mesh


class AmassDataset(BaseDataset):
    def __init__(
        self,
        motion_frames=120,
        l_factor=1.5,  # speed augmentation
        skip_moyo=True,  # not contained in the ICCV19 released version
        cam_augmentation="v11",
        random1024=False,  # DEBUG
        limit_size=None,
    ):
        self.root = Path("inputs/AMASS/hmr4d_support")
        self.motion_frames = motion_frames
        self.l_factor = l_factor
        self.random1024 = random1024
        self.skip_moyo = skip_moyo
        self.dataset_name = "AMASS"
        super().__init__(cam_augmentation, limit_size)

    def _load_dataset(self):
        filename = self.root / "smplxpose_v2.pth"
        Log.info(f"[{self.dataset_name}] Loading from {filename} ...")
        tic = Log.time()
        if self.random1024:  # Debug, faster loading
            try:
                Log.info(f"[{self.dataset_name}] Loading 1024 samples for debugging ...")
                self.motion_files = torch.load(self.root / "smplxpose_v2_random1024.pth")
            except:
                Log.info(f"[{self.dataset_name}] Not found! Saving 1024 samples for debugging ...")
                self.motion_files = torch.load(filename)
                keys = list(self.motion_files.keys())
                keys = np.random.choice(keys, 1024, replace=False)
                self.motion_files = {k: self.motion_files[k] for k in keys}
                torch.save(self.motion_files, self.root / "smplxpose_v2_random1024.pth")
        else:
            self.motion_files = torch.load(filename)
        self.seqs = list(self.motion_files.keys())
        Log.info(f"[{self.dataset_name}] {len(self.seqs)} sequences. Elapsed: {Log.time() - tic:.2f}s")

    def _get_idx2meta(self):
        # We expect to see the entire sequence during one epoch,
        # so each sequence will be sampled max(SeqLength // MotionFrames, 1) times
        seq_lengths = []
        self.idx2meta = []

        # Skip too-long idle-prefix
        motion_start_id = {}
        for vid in self.motion_files:
            if self.skip_moyo and "moyo_smplxn" in vid:
                continue
            seq_length = self.motion_files[vid]["pose"].shape[0]
            start_id = motion_start_id[vid] if vid in motion_start_id else 0
            seq_length = seq_length - start_id
            if seq_length < 25:  # Skip clips that are too short
                continue
            num_samples = max(seq_length // self.motion_frames, 1)
            seq_lengths.append(seq_length)
            self.idx2meta.extend([(vid, start_id)] * num_samples)
        hours = sum(seq_lengths) / 30 / 3600
        Log.info(f"[{self.dataset_name}] has {hours:.1f} hours motion -> Resampled to {len(self.idx2meta)} samples.")

    def _load_data(self, idx):
        """
        - Load original data
        - Augmentation: speed-augmentation to L frames
        """
        # Load original data
        mid, start_id = self.idx2meta[idx]
        raw_data = self.motion_files[mid]
        raw_len = raw_data["pose"].shape[0] - start_id
        data = {
            "body_pose": raw_data["pose"][start_id:, 3:],  # (F, 63)
            "betas": raw_data["beta"].repeat(raw_len, 1),  # (10)
            "global_orient": raw_data["pose"][start_id:, :3],  # (F, 3)
            "transl": raw_data["trans"][start_id:],  # (F, 3)
        }

        # Get {tgt_len} frames from data
        # Random select a subset with speed augmentation  [start, end)
        tgt_len = self.motion_frames
        raw_subset_len = np.random.randint(int(tgt_len / self.l_factor), int(tgt_len * self.l_factor))
        if raw_subset_len <= raw_len:
            start = np.random.randint(0, raw_len - raw_subset_len + 1)
            end = start + raw_subset_len
        else:  # interpolation will use all possible frames (results in a slow motion)
            start = 0
            end = raw_len
        data = {k: v[start:end] for k, v in data.items()}

        # Interpolation (vec + r6d)
        data_interpolated = interpolate_smpl_params(data, tgt_len)

        # AZ -> AY
        data_interpolated["global_orient"], data_interpolated["transl"], _ = get_tgtcoord_rootparam(
            data_interpolated["global_orient"],
            data_interpolated["transl"],
            tsf="az->ay",
        )

        data_interpolated["data_name"] = "amass"
        return data_interpolated


group_name = "train_datasets/pure_motion_amass"
MainStore.store(name="v11", node=builds(AmassDataset, cam_augmentation="v11"), group=group_name)
