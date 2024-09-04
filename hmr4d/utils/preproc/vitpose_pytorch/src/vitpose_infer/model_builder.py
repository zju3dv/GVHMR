import torch

# from configs.coco.ViTPose_base_coco_256x192 import model
from .builder.heads.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead

# import TopdownHeatmapSimpleHead
from .builder.backbones import ViT

# print(model)
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module

models = {
    "ViTPose_huge_coco_256x192": dict(
        type="TopDown",
        pretrained=None,
        backbone=dict(
            type="ViT",
            img_size=(256, 192),
            patch_size=16,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.55,
        ),
        keypoint_head=dict(
            type="TopdownHeatmapSimpleHead",
            in_channels=1280,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(
                final_conv_kernel=1,
            ),
            out_channels=17,
            loss_keypoint=dict(type="JointsMSELoss", use_target_weight=True),
        ),
        train_cfg=dict(),
        test_cfg=dict(),
    ),
    "ViTPose_base_coco_256x192": dict(
        type="TopDown",
        pretrained=None,
        backbone=dict(
            type="ViT",
            img_size=(256, 192),
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.3,
        ),
        keypoint_head=dict(
            type="TopdownHeatmapSimpleHead",
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(
                final_conv_kernel=1,
            ),
            out_channels=17,
            loss_keypoint=dict(type="JointsMSELoss", use_target_weight=True),
        ),
        train_cfg=dict(),
        test_cfg=dict(),
    ),
    "ViTPose_base_simple_coco_256x192": dict(
        type="TopDown",
        pretrained=None,
        backbone=dict(
            type="ViT",
            img_size=(256, 192),
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.3,
        ),
        keypoint_head=dict(
            type="TopdownHeatmapSimpleHead",
            in_channels=768,
            num_deconv_layers=0,
            num_deconv_filters=[],
            num_deconv_kernels=[],
            upsample=4,
            extra=dict(
                final_conv_kernel=3,
            ),
            out_channels=17,
            loss_keypoint=dict(type="JointsMSELoss", use_target_weight=True),
        ),
        train_cfg=dict(),
        test_cfg=dict(
            flip_test=True,
            post_process="default",
            shift_heatmap=False,
            target_type="GaussianHeatmap",
            modulate_kernel=11,
            use_udp=True,
        ),
    ),
}


def build_model(model_name, checkpoint=None):
    try:
        model = models[model_name]
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
