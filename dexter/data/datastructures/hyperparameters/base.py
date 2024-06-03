'''This file contains the base hyperparameters for training downstream models.
Extend from this class to build custom hyperparameter class'''

import random
import numpy as np
import torch

class BaseHyperParameters:
    def __init__(self):
        self.adam_epsilon = 1e-8
        self.batch_size=16
        self.warmup =0.2 
    def set_seed(self, seed:int=None):
        if seed is None:
            seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)