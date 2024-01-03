from torch import nn
from utility import tongue

LOSS_FUNCTION_MAP = { tongue.LINEAR_REGRESSION : nn.L1Loss }

def LossFunctionFactory( data = tongue.LINEAR_REGRESSION ):
    return LOSS_FUNCTION_MAP[ data ]
