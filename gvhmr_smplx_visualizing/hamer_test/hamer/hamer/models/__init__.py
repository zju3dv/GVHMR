from .mano_wrapper import MANO
from .hamer import HAMER
from .discriminator import Discriminator

from ..utils.download import cache_url
from ..configs import CACHE_DIR_HAMER


def download_models(folder=CACHE_DIR_HAMER):
    """Download checkpoints and files for running inference.
    """
    import os
    os.makedirs(folder, exist_ok=True)
    download_files = {
        "hamer_demo_data.tar.gz"      : ["https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz", folder],
    }
    
    for file_name, url in download_files.items():
        output_path = os.path.join(url[1], file_name)
        if not os.path.exists(output_path):
            print("Downloading file: " + file_name)
            # output = gdown.cached_download(url[0], output_path, fuzzy=True)
            output = cache_url(url[0], output_path)
            assert os.path.exists(output_path), f"{output} does not exist"

            # if ends with tar.gz, tar -xzf
            if file_name.endswith(".tar.gz"):
                print("Extracting file: " + file_name)
                os.system("tar --no-same-owner --warning=no-unknown-keyword -xvf " + output_path)

DEFAULT_CHECKPOINT=f'{CACHE_DIR_HAMER}/hamer_ckpts/checkpoints/hamer.ckpt'
def load_hamer(checkpoint_path=DEFAULT_CHECKPOINT):
    from pathlib import Path
    from ..configs import get_config
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()

    model = HAMER.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, init_renderer=False)
    return model, model_cfg
