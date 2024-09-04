from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import rich
import rich.tree
import rich.syntax
from hmr4d.utils.pylogger import Log


@rank_zero_only
def print_cfg(cfg: DictConfig, use_rich: bool = False):
    if use_rich:
        print_order = ("data", "model", "callbacks", "logger", "pl_trainer")
        style = "dim"
        tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

        # add fields from `print_order` to queue
        # add all the other fields to queue (not specified in `print_order`)
        queue = []
        for field in print_order:
            queue.append(field) if field in cfg else Log.warn(f"Field '{field}' not found in config. Skipping.")
        for field in cfg:
            if field not in queue:
                queue.append(field)

        # generate config tree from queue
        for field in queue:
            branch = tree.add(field, style=style, guide_style=style)
            config_group = cfg[field]
            if isinstance(config_group, DictConfig):
                branch_content = OmegaConf.to_yaml(config_group, resolve=False)
            else:
                branch_content = str(config_group)
            branch.add(rich.syntax.Syntax(branch_content, "yaml"))
        rich.print(tree)
    else:
        Log.info(OmegaConf.to_yaml(cfg, resolve=False))
