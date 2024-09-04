import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from hydra.utils import instantiate
from torch.utils.data import DataLoader, ConcatDataset, Subset
from omegaconf import ListConfig, DictConfig
from hmr4d.utils.pylogger import Log
from numpy.random import choice
from torch.utils.data import default_collate


import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def collate_fn(batch):
    """Handle meta and Add batch size to the return dict
    Args:
        batch: list of dict, each dict is a data point
    """
    # Assume all keys in the batch are the same
    return_dict = {}
    for k in batch[0].keys():
        if k.startswith("meta"):  # data information, do not batch
            return_dict[k] = [d[k] for d in batch]
        else:
            return_dict[k] = default_collate([d[k] for d in batch])
    return_dict["B"] = len(batch)
    return return_dict


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_opts: DictConfig, loader_opts: DictConfig, limit_each_trainset=None):
        """This is a general datamodule that can be used for any dataset.
        Train uses ConcatDataset
        Val and Test use CombinedLoader, sequential, completely consumes ecah iterable sequentially, and returns a triplet (data, idx, iterable_idx)

        Args:
            dataset_opts: the target of the dataset. e.g. dataset_opts.train = {_target_: ..., limit_size: None}
            loader_opts: the options for the dataset
            limit_each_trainset: limit the size of each dataset, None means no limit, useful for debugging
        """
        super().__init__()
        self.loader_opts = loader_opts
        self.limit_each_trainset = limit_each_trainset

        # Train uses concat dataset
        if "train" in dataset_opts:
            assert "train" in self.loader_opts, "train not in loader_opts"
            split_opts = dataset_opts.get("train")
            assert isinstance(split_opts, DictConfig), "split_opts should be a dict for each dataset"
            dataset = []
            dataset_num = len(split_opts)
            for idx, (k, v) in enumerate(split_opts.items()):
                dataset_i = instantiate(v)
                if self.limit_each_trainset:
                    dataset_i = Subset(dataset_i, choice(len(dataset_i), self.limit_each_trainset))
                dataset.append(dataset_i)
                Log.info(f"[Train Dataset][{idx+1}/{dataset_num}]: name={k}, size={len(dataset[-1])}, {v._target_}")
            dataset = ConcatDataset(dataset)
            self.trainset = dataset
            Log.info(f"[Train Dataset][All]: ConcatDataset size={len(dataset)}")
            Log.info(f"")

        # Val and Test use sequential dataset
        for split in ("val", "test"):
            if split not in dataset_opts:
                continue
            assert split in self.loader_opts, f"split={split} not in loader_opts"
            split_opts = dataset_opts.get(split)
            assert isinstance(split_opts, DictConfig), "split_opts should be a dict for each dataset"
            dataset = []
            dataset_num = len(split_opts)
            for idx, (k, v) in enumerate(split_opts.items()):
                dataset.append(instantiate(v))
                dataset_type = "Val Dataset" if split == "val" else "Test Dataset"
                Log.info(f"[{dataset_type}][{idx+1}/{dataset_num}]: name={k}, size={len(dataset[-1])}, {v._target_}")
            setattr(self, f"{split}sets", dataset)
            Log.info(f"")

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            return DataLoader(
                self.trainset,
                shuffle=True,
                num_workers=self.loader_opts.train.num_workers,
                persistent_workers=True and self.loader_opts.train.num_workers > 0,
                batch_size=self.loader_opts.train.batch_size,
                drop_last=True,
                collate_fn=collate_fn,
            )
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valsets"):
            loaders = []
            for valset in self.valsets:
                loaders.append(
                    DataLoader(
                        valset,
                        shuffle=False,
                        num_workers=self.loader_opts.val.num_workers,
                        persistent_workers=True and self.loader_opts.val.num_workers > 0,
                        batch_size=self.loader_opts.val.batch_size,
                        collate_fn=collate_fn,
                    )
                )
            return CombinedLoader(loaders, mode="sequential")
        else:
            return None

    def test_dataloader(self):
        if hasattr(self, "testsets"):
            loaders = []
            for testset in self.testsets:
                loaders.append(
                    DataLoader(
                        testset,
                        shuffle=False,
                        num_workers=self.loader_opts.test.num_workers,
                        persistent_workers=False,
                        batch_size=self.loader_opts.test.batch_size,
                        collate_fn=collate_fn,
                    )
                )
            return CombinedLoader(loaders, mode="sequential")
        else:
            return super().test_dataloader()
