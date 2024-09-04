import torch

# from configs.coco.ViTPose_base_coco_256x192 import model
from .heads.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead

# import TopdownHeatmapSimpleHead
from .backbones import ViT

# print(model)
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module


def build_model(model_name, checkpoint=None):
    try:
        path = ".configs.coco." + model_name
        mod = import_module(path, package="src.vitpose_infer")

        model = getattr(mod, "model")
        # from path import model
    except:
        raise ValueError("not a correct config")

    head = TopdownHeatmapSimpleHead(
        in_channels=model["keypoint_head"]["in_channels"],
        out_channels=model["keypoint_head"]["out_channels"],
        num_deconv_filters=model["keypoint_head"]["num_deconv_filters"],
        num_deconv_kernels=model["keypoint_head"]["num_deconv_kernels"],
        num_deconv_layers=model["keypoint_head"]["num_deconv_layers"],
        extra=model["keypoint_head"]["extra"],
    )
    # print(head)
    backbone = ViT(
        img_size=model["backbone"]["img_size"],
        patch_size=model["backbone"]["patch_size"],
        embed_dim=model["backbone"]["embed_dim"],
        depth=model["backbone"]["depth"],
        num_heads=model["backbone"]["num_heads"],
        ratio=model["backbone"]["ratio"],
        mlp_ratio=model["backbone"]["mlp_ratio"],
        qkv_bias=model["backbone"]["qkv_bias"],
        drop_path_rate=model["backbone"]["drop_path_rate"],
    )

    class VitPoseModel(nn.Module):
        def __init__(self, backbone, keypoint_head):
            super(VitPoseModel, self).__init__()
            self.backbone = backbone
            self.keypoint_head = keypoint_head

        def forward(self, x):
            x = self.backbone(x)
            x = self.keypoint_head(x)
            return x

    pose = VitPoseModel(backbone, head)
    if checkpoint is not None:
        check = torch.load(checkpoint)

        pose.load_state_dict(check["state_dict"])
    return pose


# pose = build_model('ViTPose_base_coco_256x192','./models/vitpose-b-multi-coco.pth')
