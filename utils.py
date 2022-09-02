import os
import random
import torch
import numpy as np
import yaml

# some constants from ViT-B_16 pretrained state dict
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

# some constants from SimpleViT
HEAD = "linear_head"

def np2th(weights, conv=False):
    """For loading from pretrained. Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_config(config_dir, config_id):
    with open(os.path.join(config_dir, f"{config_id}.YAML")) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
    
def get_cv_folds(cv, test_kfold):
    test_kfold = str(test_kfold)
    test_fold = cv[test_kfold]
    train_fold = [x for k, x in cv.items() if k!=test_kfold]
    train_fold = [x for fold in train_fold for x in fold]
    return train_fold, test_fold

class EarlyStopper():
    def __init__(self, agg=200, delta=0.005):
        """
        Stopping criteria: running median over agg steps is worse than any previous median by more than delta
        :param agg: number of steps to aggregate metric over
        :param delta: maximum change in running median before stopping
        self.current_step counts the number of val performed, not the global steps
        """
        self.history = []
        self.medians = []
        self.agg = agg
        self.delta = delta
        self.current_step = 0
    
    def step(self, v):
        self.history.append(v)
        self.current_step += 1
    
    def loss_check_stop(self):
        # stop if current median HIGHER than previous median by delta
        if self.current_step < self.agg:
            return False
        else:
            # running median in agg range
            current = np.median(self.history[self.current_step-self.agg:self.current_step])
            self.medians.append(current)
            # check if current median worse
            for m in self.medians:
                if current > (m + self.delta):
                    return True
            return False
        
    def acc_check_stop(self):
        # stop if current median LOWER than previous median by delta
        if self.current_step < self.agg:
            return False
        else:
            # running median in agg range
            current = np.median(self.history[self.current_step-self.agg:self.current_step])
            self.medians.append(current)
            # check if current median worse 
            for m in self.medians:
                if current < (m - self.delta):
                    return True
            return False

