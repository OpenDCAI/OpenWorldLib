import math
import torch


class MultiCosineAnnealingLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, max_steps, warmup_steps=0, eta_min_ratio=0.1, last_step=0):
        """
        Args:
            optimizer: model optimizer, must have the property of adjustable learning rate
            max_steps: number of learning rate annealing cycles, indicating the number of training steps required from the initial learning rate to eta_min
            eta_min: minimum learning rate, default is 0
            last_step: the step of the last training, default is 0, indicating starting from the beginning
        """
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.eta_mins = [base_lr * eta_min_ratio for base_lr in self.base_lrs]
        self.last_step = last_step
        self.current_lrs = self.base_lrs
        self.last_lrs = [0] * len(self.base_lrs)

    def get_lr(self):
        """
        Calculate the current learning rate
        """
        step = self.last_step + 1
        if step < self.warmup_steps:
            return self.base_lrs
        elif step >= self.max_steps:
            return self.eta_mins
        else:
            lrs = []
            for i, base_lr in enumerate(self.base_lrs):
                lr = self.eta_mins[i] + (base_lr - self.eta_mins[i]) * (1 + math.cos(math.pi * (step - self.warmup_steps) / (self.max_steps - self.warmup_steps))) / 2
                lrs.append(lr)
            return lrs

    def step(self):
        """
        Update the learning rate for each step
        """
        self.last_lrs = self.current_lrs
        self.last_step += 1
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
        self.current_lrs = lrs

    def get_last_lr(self):
        return self.last_lrs
    

class MultiLinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, max_steps, warmup_steps=0, eta_min_ratio=0.1, last_step=0):
        """
        Args:
            optimizer: model optimizer
            max_steps: total training steps (from base lr decay到 eta_min)
            warmup_steps: warmup阶段的步数
            eta_min_ratio: 最小学习率 = base_lr * eta_min_ratio
            last_step: 从哪个step开始
        """
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.eta_mins = [base_lr * eta_min_ratio for base_lr in self.base_lrs]
        self.last_step = last_step
        self.current_lrs = self.base_lrs
        self.last_lrs = [0] * len(self.base_lrs)

    def get_lr(self):
        step = self.last_step + 1

        # Warmup phase: linear growth
        if step <= self.warmup_steps and self.warmup_steps > 0:
            lrs = []
            for base_lr in self.base_lrs:
                lr = base_lr * step / self.warmup_steps
                lrs.append(lr)
            return lrs

        # Cosine Annealing phase
        elif step < self.max_steps:
            lrs = []
            for i, base_lr in enumerate(self.base_lrs):
                lr = self.eta_mins[i] + (base_lr - self.eta_mins[i]) * \
                     (1 + math.cos(math.pi * (step - self.warmup_steps) /
                                   (self.max_steps - self.warmup_steps))) / 2
                lrs.append(lr)
            return lrs

        # Final phase: fixed at eta_min
        else:
            return self.eta_mins

    def step(self):
        self.last_lrs = self.current_lrs
        self.last_step += 1
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
        self.current_lrs = lrs

    def get_last_lr(self):
        return self.last_lrs