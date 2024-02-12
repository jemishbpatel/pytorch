import torch
LINEAR_REGRESSION = "LinearRegression"
LINEAR_REGRESSION_VERSION1 = "V1"
LINEAR_REGRESSION_VERSION2 = "V2"
CIRCLE_MODEL_VERSION1 = "V1"
CIRCLE_MODEL_VERSION1 = "V2"
BINARY_CLASSIFICATION = "BinaryClassifiction"
MULTILCLASS_CLASSIFICATION = "MulticlassClassification"
COMPUTER_VISION_MODEL = "ComputerVisionModel"
LINEAR_MODEL_TYPE = "Linear"
NON_LINEAR_MODEL_TYPE = "Non-linear"
TRANSFER_LEARNING = "transfer-learning"

TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
