import torch
from utility import tongue

OPTIMIZER_MAP = { tongue.LINEAR_REGRESSION: torch.optim.SGD }

def OptmizerFactory( data = tongue.LINEAR_REGRESSION ):
    return OPTIMIZER_MAP[ data ]
