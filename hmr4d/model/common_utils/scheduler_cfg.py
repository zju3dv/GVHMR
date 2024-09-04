from omegaconf import DictConfig, ListConfig
from hmr4d.configs import MainStore, builds

# do not perform scheduling
default = DictConfig({"scheduler": None})
MainStore.store(name="default", node=default, group=f"scheduler_cfg")


# epoch-based
def epoch_half_by(milestones=[100, 200, 300]):
    return DictConfig(
        {
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.MultiStepLR",
                "milestones": milestones,
                "gamma": 0.5,
            },
            "interval": "epoch",
            "frequency": 1,
        }
    )


MainStore.store(name="epoch_half_100_200_300", node=epoch_half_by([100, 200, 300]), group=f"scheduler_cfg")
MainStore.store(name="epoch_half_100_200", node=epoch_half_by([100, 200]), group=f"scheduler_cfg")
MainStore.store(name="epoch_half_200_350", node=epoch_half_by([200, 350]), group=f"scheduler_cfg")
MainStore.store(name="epoch_half_300", node=epoch_half_by([300]), group=f"scheduler_cfg")


# epoch-based
def warmup_epoch_half_by(warmup=10, milestones=[100, 200, 300]):
    return DictConfig(
        {
            "scheduler": {
                "_target_": "hmr4d.model.common_utils.scheduler.WarmupMultiStepLR",
                "milestones": milestones,
                "warmup": warmup,
                "gamma": 0.5,
            },
            "interval": "epoch",
            "frequency": 1,
        }
    )


MainStore.store(name="warmup_5_epoch_half_200_350", node=warmup_epoch_half_by(5, [200, 350]), group=f"scheduler_cfg")
MainStore.store(name="warmup_10_epoch_half_200_350", node=warmup_epoch_half_by(10, [200, 350]), group=f"scheduler_cfg")
