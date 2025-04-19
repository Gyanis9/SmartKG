import random

import torch

from config import get_config
from trainer import IncrementalTrainer

if __name__ == "__main__":
    config = get_config()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    trainer = IncrementalTrainer(config)
    trainer.run()
