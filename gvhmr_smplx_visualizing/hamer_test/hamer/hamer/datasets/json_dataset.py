import copy
import os
import json
import glob
import numpy as np
import torch
from typing import Any, Dict, List
from yacs.config import CfgNode
import braceexpand
import cv2

from .dataset import Dataset
from .utils import get_example, expand_to_aspect_ratio
from .smplh_prob_filter import poses_check_probable, load_amass_hist_smooth

def expand(s):
    return os.path.expanduser(os.path.expandvars(s))
def expand_urls(urls: str|List[str]):
    if isinstance(urls, str):
        urls = [urls]
    urls = [u for url in urls for u in braceexpand.braceexpand(expand(url))]
    return urls

AIC_TRAIN_CORRUPT_KEYS = {
    '0a047f0124ae48f8eee15a9506ce1449ee1ba669',
    '1a703aa174450c02fbc9cfbf578a5435ef403689',
    '0394e6dc4df78042929b891dbc24f0fd7ffb6b6d',
    '5c032b9626e410441544c7669123ecc4ae077058',
    'ca018a7b4c5f53494006ebeeff9b4c0917a55f07',
    '4a77adb695bef75a5d34c04d589baf646fe2ba35',
    'a0689017b1065c664daef4ae2d14ea03d543217e',
    '39596a45cbd21bed4a5f9c2342505532f8ec5cbb',
    '3d33283b40610d87db660b62982f797d50a7366b',
}
CORRUPT_KEYS = {
    *{f'aic-train/{k}' for k in AIC_TRAIN_CORRUPT_KEYS},
    *{f'aic-train-vitpose/{k}' for k in AIC_TRAIN_CORRUPT_KEYS},
}

FLIP_KEYPOINT_PERMUTATION = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])
DEFAULT_IMG_SIZE = 256

class JsonDataset(Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 dataset_file: str,
                 img_dir: str,
                 right: bool,
                 train: bool = False,
                 prune: Dict[str, Any] = {},
                 **kwargs):
        """
        Dataset class used for loading images and corresponding annotations.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        super(JsonDataset, self).__init__()
        self.train = train
        self.cfg = cfg

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        self.img_dir = img_dir
        boxes = np.array(json.load(open(dataset_file, 'rb')))

        self.imgname = glob.glob(os.path.join(self.img_dir,'*.jpg'))
        self.imgname.sort()

        self.flip_keypoint_permutation = copy.copy(FLIP_KEYPOINT_PERMUTATION)

        num_pose = 3 * (self.cfg.MANO.NUM_HAND_JOINTS + 1)

        # Bounding boxes are assumed to be in the center and scale format
        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = 2 * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)
        if right:
            self.right = np.ones(len(self.imgname), dtype=np.float32)
        else:
            self.right = np.zeros(len(self.imgname), dtype=np.float32)
        assert self.scale.shape == (len(self.center), 2)

        # Get gt SMPLX parameters, if available
        try:
            self.hand_pose = self.data['hand_pose'].astype(np.float32)
            self.has_hand_pose = self.data['has_hand_pose'].astype(np.float32)
        except:
            self.hand_pose = np.zeros((len(self.imgname), num_pose), dtype=np.float32)
            self.has_hand_pose = np.zeros(len(self.imgname), dtype=np.float32)
        try:
            self.betas = self.data['betas'].astype(np.float32)
            self.has_betas = self.data['has_betas'].astype(np.float32)
        except:
            self.betas = np.zeros((len(self.imgname), 10), dtype=np.float32)
            self.has_betas = np.zeros(len(self.imgname), dtype=np.float32)

        # Try to get 2d keypoints, if available
        try:
            hand_keypoints_2d = self.data['hand_keypoints_2d']
        except:
            hand_keypoints_2d = np.zeros((len(self.center), 21, 3))
        ## Try to get extra 2d keypoints, if available
        #try:
        #    extra_keypoints_2d = self.data['extra_keypoints_2d']
        #except KeyError:
        #    extra_keypoints_2d = np.zeros((len(self.center), 19, 3))

        #self.keypoints_2d = np.concatenate((hand_keypoints_2d, extra_keypoints_2d), axis=1).astype(np.float32)
        self.keypoints_2d = hand_keypoints_2d

        # Try to get 3d keypoints, if available
        try:
            hand_keypoints_3d = self.data['hand_keypoints_3d'].astype(np.float32)
        except:
            hand_keypoints_3d = np.zeros((len(self.center), 21, 4), dtype=np.float32)
        ## Try to get extra 3d keypoints, if available
        #try:
        #    extra_keypoints_3d = self.data['extra_keypoints_3d'].astype(np.float32)
        #except KeyError:
        #    extra_keypoints_3d = np.zeros((len(self.center), 19, 4), dtype=np.float32)

        self.keypoints_3d = hand_keypoints_3d

        #body_keypoints_3d[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], -1] = 0

        #self.keypoints_3d = np.concatenate((body_keypoints_3d, extra_keypoints_3d), axis=1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.scale)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns an example from the dataset.
        """
        try:
            image_file = self.imgname[idx].decode('utf-8')
        except AttributeError:
            image_file = self.imgname[idx]
        keypoints_2d = self.keypoints_2d[idx].copy()
        keypoints_3d = self.keypoints_3d[idx].copy()

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]
        scale = self.scale[idx]
        right = self.right[idx].copy()
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        #bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()
        bbox_size = ((scale*200).max())
        bbox_expand_factor = bbox_size / ((scale*200).max())
        hand_pose = self.hand_pose[idx].copy().astype(np.float32)
        betas = self.betas[idx].copy().astype(np.float32)

        has_hand_pose = self.has_hand_pose[idx].copy()
        has_betas = self.has_betas[idx].copy()

        mano_params = {'global_orient': hand_pose[:3],
                       'hand_pose': hand_pose[3:],
                       'betas': betas
                      }

        has_mano_params = {'global_orient': has_hand_pose,
                           'hand_pose': has_hand_pose,
                           'betas': has_betas
                           }

        mano_params_is_axis_angle = {'global_orient': True,
                                     'hand_pose': True,
                                     'betas': False
                                    }

        augm_config = self.cfg.DATASETS.CONFIG
        # Crop image and (possibly) perform data augmentation
        img_patch, keypoints_2d, keypoints_3d, mano_params, has_mano_params, img_size = get_example(image_file,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    keypoints_2d, keypoints_3d,
                                                                                                    mano_params, has_mano_params,
                                                                                                    self.flip_keypoint_permutation,
                                                                                                    self.img_size, self.img_size,
                                                                                                    self.mean, self.std, self.train, right, augm_config)

        item = {}
        # These are the keypoints in the original image coordinates (before cropping)
        orig_keypoints_2d = self.keypoints_2d[idx].copy()

        item['img'] = img_patch
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['keypoints_3d'] = keypoints_3d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = self.center[idx].copy()
        item['box_size'] = bbox_size
        item['bbox_expand_factor'] = bbox_expand_factor
        item['img_size'] = 1.0 * img_size[::-1].copy()
        item['mano_params'] = mano_params
        item['has_mano_params'] = has_mano_params
        item['mano_params_is_axis_angle'] = mano_params_is_axis_angle
        item['imgname'] = image_file
        item['personid'] = int(self.personid[idx])
        item['idx'] = idx
        item['_scale'] = scale
        item['right'] = self.right[idx].copy()
        return item
