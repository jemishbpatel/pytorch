from torch import nn
from utility import tongue

LOSS_FUNCTION_MAP = { tongue.LINEAR_REGRESSION : nn.L1Loss,
                      tongue.BINARY_CLASSIFICATION : nn.BCEWithLogitsLoss }

def LossFunctionFactory( data = tongue.LINEAR_REGRESSION ):
    return LOSS_FUNCTION_MAP[ data ]
