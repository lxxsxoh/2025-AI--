import os
import math
import logging
from torch.optim.lr_scheduler import LambdaLR


class CosineDecayWithWarmup(LambdaLR):
    def __init__(self, optimizer, warmup_epochs, num_epochs):
        def lr_lambda(cur_step):
            if cur_step < warmup_epochs:
                return float(cur_step) / float(max(1.0, warmup_epochs))
            progress = float(cur_step - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        self.warmup_epochs = warmup_epochs
        self.num_epochs = num_epochs
        super().__init__(optimizer, lr_lambda, last_epoch=-1)

    def __str__(self):
        return f"CosineDecayWithWarmup(warmup_epochs={self.warmup_epochs}, num_epochs={self.num_epochs})"


class AverageMeter:
    def __init__(self):
        self.history = []
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def step(self):
        if self.count > 0:
            self.history.append(self.sum / self.count)
        self.reset()

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    def __list__(self):
        return self.history


def get_logger(result_path):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s - %(levelname)s] : %(message)s")
        file_handler = logging.FileHandler(os.path.join(result_path, "train.log"))
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
