import torch
LINEAR_REGRESSION = "LinearRegression"
LINEAR_REGRESSION_VERSION1 = "V1"
LINEAR_REGRESSION_VERSION2 = "V2"

TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
