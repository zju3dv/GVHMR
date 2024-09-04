import torch
from torch.utils import data
import numpy as np
from pathlib import Path
from hmr4d.utils.pylogger import Log


class ImgfeatMotionDatasetBase(data.Dataset):
    def __init__(self):
        super().__init__()
        self._load_dataset()
        self._get_idx2meta()  # -> Set self.idx2meta

    def __len__(self):
        return len(self.idx2meta)

    def _load_dataset(self):
        raise NotImplemented

    def _get_idx2meta(self):
        raise NotImplemented

    def _load_data(self, idx):
        raise NotImplemented

    def _process_data(self, data, idx):
        raise NotImplemented

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data
