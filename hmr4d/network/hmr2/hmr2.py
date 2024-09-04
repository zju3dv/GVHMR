import torch
import pytorch_lightning as pl
from yacs.config import CfgNode
from .vit import ViT
from .smpl_head import SMPLTransformerDecoderHead

from pytorch3d.transforms import matrix_to_axis_angle
from hmr4d.utils.geo.hmr_cam import compute_transl_full_cam


class HMR2(pl.LightningModule):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg
        self.backbone = ViT(
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
        )
        self.smpl_head = SMPLTransformerDecoderHead(cfg)

    def forward(self, batch, feat_mode=True):
        """this file has been modified
        Args:
            feat_mode: default True, as we only need the feature token output for the HMR4D project;
                       when False, the full process of HMR2 will be executed.
        """
        # Backbone
        x = batch["img"][:, :, :, 32:-32]
        vit_feats = self.backbone(x)

        # Output head
        if feat_mode:
            token_out = self.smpl_head(vit_feats, only_return_token_out=True)  # (B, 1024)
            return token_out

        # return full process
        pred_smpl_params, pred_cam, _, token_out = self.smpl_head(vit_feats, only_return_token_out=False)
        output = {}
        output["token_out"] = token_out
        output["smpl_params"] = {
            "body_pose": matrix_to_axis_angle(pred_smpl_params["body_pose"]).flatten(-2),  # (B, 23, 3)
            "betas": pred_smpl_params["betas"],  # (B, 10)
            "global_orient": matrix_to_axis_angle(pred_smpl_params["global_orient"])[:, 0],  # (B, 3)
            "transl": compute_transl_full_cam(pred_cam, batch["bbx_xys"], batch["K_fullimg"]),  # (B, 3)
        }

        return output
