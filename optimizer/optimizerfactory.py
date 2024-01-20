import torch
from torch import nn
from utility import tongue

OPTIMIZER_MAP = { tongue.LINEAR_REGRESSION: torch.optim.SGD,
                  tongue.BINARY_CLASSIFICATION : torch.optim.SGD,
                  tongue.MULTILCLASS_CLASSIFICATION : torch.optim.SGD,
                  tongue.COMPUTER_VISION_MODEL : torch.optim.SGD }

def OptmizerFactory( data = tongue.LINEAR_REGRESSION ):
    return OPTIMIZER_MAP[ data ]
