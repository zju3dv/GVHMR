import torch
from bisect import bisect_right


class WarmupMultiStepLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, milestones, warmup=0, gamma=0.1, last_epoch=-1, verbose="deprecated"):
        """Assume optimizer does not change lr; Scheduler is called epoch-based"""
        self.milestones = milestones
        self.warmup = warmup
        assert warmup < milestones[0]
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        base_lrs = self.base_lrs  # base lr for each groups
        n_groups = len(base_lrs)
        comming_epoch = self.last_epoch  # the lr will be set for the comming epoch, starts from 0

        # add extra warmup
        if comming_epoch < self.warmup:
            # e.g. comming_epoch [0, 1, 2] for warmup == 3
            # lr should be base_lr * (last_epoch+1) / (warmup + 1), e.g. [0.25, 0.5, 0.75] * base_lr
            lr_factor = (self.last_epoch + 1) / (self.warmup + 1)
            return [base_lrs[i] * lr_factor for i in range(n_groups)]
        else:
            # bisect_right([3,5,7], 0) -> 0; bisect_right([3,5,7], 5) -> 2
            p = bisect_right(self.milestones, comming_epoch)
            lr_factor = self.gamma**p
            return [base_lrs[i] * lr_factor for i in range(n_groups)]
