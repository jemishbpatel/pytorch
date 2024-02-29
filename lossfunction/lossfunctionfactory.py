from torch import nn
from utility import tongue

LOSS_FUNCTION_MAP = { tongue.LINEAR_REGRESSION : nn.L1Loss,
                      tongue.BINARY_CLASSIFICATION : nn.BCEWithLogitsLoss,
                      tongue.MULTILCLASS_CLASSIFICATION : nn.CrossEntropyLoss,
                      tongue.COMPUTER_VISION_MODEL : nn.CrossEntropyLoss,
                      tongue.EFFICIENTNET_B2_MODEL : nn.CrossEntropyLoss }

def LossFunctionFactory( data = tongue.LINEAR_REGRESSION ):
    return LOSS_FUNCTION_MAP[ data ]
