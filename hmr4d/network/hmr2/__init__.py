import torch
from .hmr2 import HMR2
from pathlib import Path
from .configs import get_config
from hmr4d import PROJ_ROOT

HMR2A_CKPT = PROJ_ROOT / f"inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"  # this is HMR2.0a, follow WHAM


def load_hmr2(checkpoint_path=HMR2A_CKPT):
    model_cfg = str((Path(__file__).parent / "configs/model_config.yaml").resolve())
    model_cfg = get_config(model_cfg)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        assert (
            model_cfg.MODEL.IMAGE_SIZE == 256
        ), f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]  # (W, H)
        model_cfg.freeze()

    # Setup model and Load weights.
    # model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    model = HMR2(model_cfg)

    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    keys = [k for k in state_dict.keys() if k.split(".")[0] in ["backbone", "smpl_head"]]
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    model.load_state_dict(state_dict, strict=True)

    return model
