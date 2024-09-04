import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from time import time
from collections import deque

from hmr4d.configs import MainStore, builds


class TrainSpeedTimer(pl.Callback):
    def __init__(self, N_avg=5):
        """
        This callback times the training speed (averge over recent 5 iterations)
            1. Data waiting time: this should be small, otherwise the data loading should be improved
            2. Single batch time: this is the time for one batch of training (excluding data waiting)
        """
        super().__init__()
        self.last_batch_end = None
        self.this_batch_start = None

        # time queues for averaging
        self.data_waiting_time_queue = deque(maxlen=N_avg)
        self.single_batch_time_queue = deque(maxlen=N_avg)

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Count the time of data waiting"""
        if self.last_batch_end is not None:
            # This should be small, otherwise the data loading should be improved
            data_waiting = time() - self.last_batch_end

            # Average the time
            self.data_waiting_time_queue.append(data_waiting)
            average_time = sum(self.data_waiting_time_queue) / len(self.data_waiting_time_queue)

            # Log to prog-bar
            pl_module.log(
                "train_timer/data_waiting", average_time, on_step=True, on_epoch=False, prog_bar=True, logger=True
            )

        self.this_batch_start = time()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Effective training time elapsed (excluding data waiting)
        single_batch = time() - self.this_batch_start

        # Average the time
        self.single_batch_time_queue.append(single_batch)
        average_time = sum(self.single_batch_time_queue) / len(self.single_batch_time_queue)

        # Log iter time
        pl_module.log(
            "train_timer/single_batch", average_time, on_step=True, on_epoch=False, prog_bar=False, logger=True
        )

        # Set timer for counting data waiting
        self.last_batch_end = time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        # Reset the timer
        self.last_batch_end = None
        self.this_batch_start = None
        # Clear the queue
        self.data_waiting_time_queue.clear()
        self.single_batch_time_queue.clear()


group_name = "callbacks/train_speed_timer"
base = builds(TrainSpeedTimer, populate_full_signature=True)
MainStore.store(name="base", node=base, group=group_name)
