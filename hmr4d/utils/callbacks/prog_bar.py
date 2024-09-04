from collections import OrderedDict
from numbers import Number
from datetime import datetime, timedelta
from typing import Any, Dict, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar, Tqdm, convert_inf
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as pl

from hmr4d.utils.pylogger import Log
from time import time
from collections import deque
import sys
from hmr4d.configs import MainStore, builds

# ========== Helper functions ========== #


def format_num(n):
    f = "{0:.3g}".format(n).replace("+0", "+").replace("-0", "-")
    n = str(n)
    return f if len(f) < len(n) else n


def convert_kwargs_to_str(**kwargs):
    # Sort in alphabetical order to be more deterministic
    postfix = OrderedDict([])
    for key in sorted(kwargs.keys()):
        new_key = key.split("/")[-1]
        postfix[new_key] = kwargs[key]
    # Preprocess stats according to datatype
    for key in postfix.keys():
        # Number: limit the length of the string
        if isinstance(postfix[key], Number):
            postfix[key] = format_num(postfix[key])
        # Else for any other type, try to get the string conversion
        elif not isinstance(postfix[key], str):
            postfix[key] = str(postfix[key])
        # Else if it's a string, don't need to preprocess anything
    # Stitch together to get the final postfix
    postfix = ", ".join(key + "=" + postfix[key].strip() for key in postfix.keys())
    return postfix


def convert_t_to_str(t):
    """Convert time in second to string in format hour:minute:second.
    If hour is 0, don't show it. Always show minute and second.
    """
    t_str = timedelta(seconds=t)  # e.g. 0:00:00.704186
    t_str = str(t_str).split(".")[0]  # e.g. 0:00:00
    if t_str[:2] == "0:":
        t_str = t_str[2:]
    return t_str


class MyTQDMProgressBar(TQDMProgressBar, pl.Callback):
    def init_train_tqdm(self):
        bar = Tqdm(
            desc="Training",  # this will be overwritten anyway
            bar_format="{desc}{percentage:3.0f}%[{bar:10}][{n_fmt}/{total_fmt}, {elapsed}→{remaining},{rate_fmt}]{postfix}",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            smoothing=0,
            dynamic_ncols=False,
        )
        return bar

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # this function also updates the main progress bar
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        # in this function, we only set the postfix of the main progress bar
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            # Set post-fix string
            # 1. maximum GPU usage
            max_mem = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
            post_fix_str = f"maxGPU={max_mem:.1f}GB"

            # 2. training metrics
            training_metrics = self.get_metrics(trainer, pl_module)
            training_metrics.pop("v_num", None)
            post_fix_str += ", " + convert_kwargs_to_str(**training_metrics)

            # extra message if applicable
            if "message" in outputs:
                post_fix_str += ", " + outputs["message"]

            self.train_progress_bar.set_postfix_str(post_fix_str)


class ProgressReporter(ProgressBar, pl.Callback):
    def __init__(
        self,
        log_every_percent: float = 0.1,  # report interval
        exp_name=None,  # if None, use pl_module.exp_name or "Unnamed Experiment"
        data_name=None,  # if None, use pl_module.exp_name or "Unknown Data"
        **kwargs,
    ):
        super().__init__()
        self.enable = True
        # 1. Store experiment meta data.
        self.log_every_percent = log_every_percent
        self.exp_name = exp_name
        self.data_name = data_name
        self.batch_time_queue = deque(maxlen=5)
        self.start_prompt = "🚀"
        self.finish_prompt = "✅"
        # 2. Utils for evaluation
        self.n_finished = 0

    def disable(self):
        self.enable = False

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        # Connect to the trainer object.
        super().setup(trainer, pl_module, stage)
        self.stage = stage
        self.time_exp_start = time()
        self.epoch_exp_start = trainer.current_epoch

        if self.exp_name is None:
            if hasattr(pl_module, "exp_name"):
                self.exp_name = pl_module.exp_name
            else:
                self.exp_name = "Unnamed Experiment"
        if self.data_name is None:
            if hasattr(pl_module, "data_name"):
                self.data_name = pl_module.data_name
            else:
                self.data_name = "Unknown Data"

    def print(self, *args: Any, **kwargs: Any) -> None:
        print(*args)

    def get_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> Dict[str, Union[str, float]]:
        """Get metrics from trainer for progress bar."""
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items

    def _should_update(self, n_finished: int, total: int) -> bool:
        """
        Rule: Log every `log_every_percent` percent, or the last batch.
        """
        log_interval = max(int(total * self.log_every_percent), 1)
        able = n_finished % log_interval == 0 or n_finished == total
        if log_interval > 10:
            able = able or n_finished in [5, 10]  # always log
        able = able and self.enable
        return able

    @rank_zero_only
    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        self.print("=" * 80)
        Log.info(
            f"{self.start_prompt}[FIT][Epoch {trainer.current_epoch}] Data: {self.data_name} Experiment: {self.exp_name}"
        )
        self.time_train_epoch_start = time()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)
        total = self.total_train_batches

        # Speed
        n_finished = batch_idx + 1
        percent = 100 * n_finished / total
        time_current = time()
        self.batch_time_queue.append(time_current)
        time_elapsed = time_current - self.time_train_epoch_start  # second
        time_remaining = time_elapsed * (total - n_finished) / n_finished  # second
        if len(self.batch_time_queue) == 1:  # cannot compute speed
            speed = 1 / time_elapsed
        else:
            speed = (len(self.batch_time_queue) - 1) / (self.batch_time_queue[-1] - self.batch_time_queue[0])

        # Skip if not update
        if not self._should_update(n_finished, total):
            return

        # ===== Set Prefix string ===== #
        # General
        desc = f"[Train]"

        # Speed: Get elapsed time and estimated remaining time
        time_elapsed_str = convert_t_to_str(time_elapsed)
        time_remaining_str = convert_t_to_str(time_remaining)
        speed_str = f"{speed:.2f}it/s" if speed > 1 else f"{1/speed:.1f}s/it"
        n_digit = len(str(total))
        desc_speed = (
            f"[{n_finished:{n_digit}d}/{total}={percent:3.0f}%, {time_elapsed_str} → {time_remaining_str}, {speed_str}]"
        )

        # ===== Set postfix string ===== #
        # 1. maximum GPU usage
        max_mem = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
        post_fix_str = f"maxGPU={max_mem:.1f}GB"

        # 2. training step metrics
        train_metrics = self.get_metrics(trainer, pl_module)
        train_metrics = {k: v for k, v in train_metrics.items() if ("train" in k and "epoch" not in k)}
        post_fix_str += ", " + convert_kwargs_to_str(**train_metrics)

        # extra message if applicable
        if "message" in outputs:
            post_fix_str += ", " + outputs["message"]
        post_fix_str = f"[{post_fix_str}]"

        # ===== Output ===== #
        bar_output = f"{desc}{desc_speed}{post_fix_str}"
        self.print(bar_output)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)

        # Clear
        self.batch_time_queue.clear()

        # Estimate Epoch time
        n_finished = trainer.current_epoch + 1 - self.epoch_exp_start
        n_to_finish = trainer.max_epochs - trainer.current_epoch - 1
        time_current = time()
        time_elapsed = time_current - self.time_exp_start
        time_remaining = time_elapsed * n_to_finish / n_finished
        time_elapsed_str = convert_t_to_str(time_elapsed)
        time_remaining_str = convert_t_to_str(time_remaining)

        # Metrics
        # training epoch metrics
        train_metrics = self.get_metrics(trainer, pl_module)
        train_metrics = {k: v for k, v in train_metrics.items() if ("train" in k and "epoch" in k)}
        train_metrics_str = convert_kwargs_to_str(**train_metrics)

        Log.info(
            f"{self.finish_prompt}[FIT][Epoch {trainer.current_epoch}] finished! {time_elapsed_str}→{time_remaining_str} | {train_metrics_str}"
        )

    # ===== Validation/Test/Prediction ===== #
    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        self.time_val_epoch_start = time()

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.n_finished += 1
        n_finished = self.n_finished
        total = self.total_val_batches
        if not self._should_update(n_finished, total):
            return

        # General
        desc = f"[Val]"

        # Speed
        percent = 100 * n_finished / total
        time_current = time()
        time_elapsed = time_current - self.time_val_epoch_start  # second
        time_remaining = time_elapsed * (total - n_finished) / n_finished  # second
        time_elapsed_str = convert_t_to_str(time_elapsed)
        time_remaining_str = convert_t_to_str(time_remaining)
        desc_speed = f"[{n_finished}/{total} ={percent:3.0f}%, {time_elapsed_str}→{time_remaining_str}]"

        # Output
        bar_output = f"{desc} {desc_speed}"
        self.print(bar_output)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Reset
        self.n_finished = 0


class EmojiProgressReporter(ProgressBar, pl.Callback):
    def __init__(
        self,
        refresh_rate_batch: Union[int, None] = 1,  # report interval of batch, set None to disable it
        refresh_rate_epoch: int = 1,  # report interval of epoch
        **kwargs,
    ):
        super().__init__()
        self.enable = True
        # Store experiment meta data.
        self.refresh_rate_batch = refresh_rate_batch
        self.refresh_rate_epoch = refresh_rate_epoch

        # Style of the progress bar.
        self.title_prompt = "📝"
        self.prog_prompt = "🚀"
        self.timer_prompt = "⌛️"
        self.metric_prompt = "📌"
        self.finish_prompt = "✅"

    def disable(self):
        self.enable = False

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        # Connect to the trainer object.
        super().setup(trainer, pl_module, stage)
        self.stage = stage
        self.time_start_batch = None
        self.time_start_epoch = None
        if hasattr(pl_module, "exp_name"):
            self.exp_name = pl_module.exp_name
        else:
            self.exp_name = "Unnamed Experiment"
            Log.warn("Experiment name not found, please set it to `pl_module.exp_name`!")

    def print(self, *args: Any, **kwargs: Any):
        print(*args)

    def get_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> Dict[str, Union[str, float]]:
        """Get metrics from trainer for progress bar."""
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return dict(sorted(items.items()))

    def _should_log_batch(self, n: int) -> bool:
        # Disable batch log.
        if self.refresh_rate_batch is None:
            return False
        # Log at the first & last batch, and every `self.refresh_rate_batch` batches.
        able = n % self.refresh_rate_batch == 0 or n == self.total_train_batches - 1
        able = able and self.enable
        return able

    def _should_log_epoch(self, n: int) -> bool:
        # Log at the first & last epoch, and every `self.refresh_rate_epoch` epochs.
        able = n % self.refresh_rate_epoch == 0 or n == self.trainer.max_epochs - 1
        able = able and self.enable
        return able

    def timestamp_delta_to_str(self, timestamp_delta: float):
        """Convert delta timestamp to string."""
        time_rest = timedelta(seconds=timestamp_delta)
        hours, remainder = divmod(time_rest.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = ""

        # Check if the time is valid. Note that, if `hours` is visible, then `minutes` must be visible.
        if hours <= 0:
            hours = None
            if minutes <= 0:
                minutes = None
                if seconds <= 0:
                    seconds = None

        time_str += f"{hours}h " if hours is not None else ""
        time_str += f"{minutes}m " if minutes is not None else ""
        time_str += f"{seconds}s" if seconds is not None else ""
        return time_str

    @rank_zero_only
    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        # Initialize some meta data.
        if self.time_start_batch is None:
            self.time_start_batch = datetime.now().timestamp()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)
        # Get some meta data.
        epoch_idx = trainer.current_epoch
        percent = 100 * (batch_idx + 1) / (self.total_train_batches + 1)
        metrics = self.get_metrics(trainer, pl_module)

        # Current time.
        time_cur_stamp = datetime.now().timestamp()
        time_cur_str = datetime.fromtimestamp(time_cur_stamp).strftime("%m-%d %H:%M:%S")
        # Rest time.
        time_rest_stamp = (time_cur_stamp - self.time_start_batch) * (100 - percent) / percent
        time_rest_str = self.timestamp_delta_to_str(time_rest_stamp)

        if not self._should_log_batch(batch_idx):
            return

        # Print the logs.
        self.print(f"{self.title_prompt} [{self.stage.upper()}] Exp: {self.exp_name}...")
        self.print(
            f"{self.prog_prompt} Ep {epoch_idx}: {int(percent):02d}% <= [{batch_idx}/{self.total_train_batches}]"
        )
        self.print(f"{self.timer_prompt} Time: {time_cur_str} | Ep Rest: {time_rest_str}")
        for k, v in metrics.items():
            self.print(f"{self.metric_prompt} {k}: {v}")
        self.print("")  # Add a blank line.

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        super().on_train_epoch_start(trainer, pl_module)
        # Initialize some meta data.
        self.time_start_batch = None
        if self.time_start_epoch is None:
            self.time_start_epoch = datetime.now().timestamp()

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        super().on_train_epoch_end(trainer, pl_module)
        # Get some meta data.
        epoch_idx = trainer.current_epoch
        percent = 100 * (epoch_idx + 1) / (self.trainer.max_epochs + 1)
        metrics = self.get_metrics(trainer, pl_module)

        # Current time.
        time_cur = datetime.now().timestamp()
        time_str = datetime.fromtimestamp(time_cur).strftime("%m-%d %H: %M:%S")
        # Rest time.
        time_rest_stamp = (time_cur - self.time_start_epoch) * (100 - percent) / percent
        time_rest_str = self.timestamp_delta_to_str(time_rest_stamp)

        if not self._should_log_batch(epoch_idx):
            return

        # Print the logs.
        self.print(f">> >> >> >>")
        self.print(f"{self.title_prompt} [{self.stage.upper()}] Exp: {self.exp_name}")
        self.print(f"{self.finish_prompt} Ep {epoch_idx} finished!")
        self.print(f"{self.timer_prompt} Time: {time_str} | Rest: {time_rest_str}")
        for k, v in metrics.items():
            self.print(f"{self.metric_prompt} {k}: {v}")
        self.print(f"<< << << <<")
        self.print("")  # Add a blank line.


group_name = "callbacks/prog_bar"
prog_reporter_base = builds(
    ProgressReporter,
    log_every_percent=0.1,
    exp_name="${exp_name}",
    data_name="${data_name}",
    populate_full_signature=True,
)
MainStore.store(name="prog_reporter_every0.1", node=prog_reporter_base, group=group_name)
MainStore.store(name="prog_reporter_every0.2", node=prog_reporter_base(log_every_percent=0.2), group=group_name)
