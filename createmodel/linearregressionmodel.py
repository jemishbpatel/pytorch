import torch
from torch import nn
from utility import tongue

class LinearRegressionModel( nn.Module ):
    def __init__( self, modelVersion = tongue.LINEAR_REGRESSION_VERSION1 ):
        super().__init__()
        self._modelVersion = modelVersion

        if self._modelVersion == tongue.LINEAR_REGRESSION_VERSION1:
            self.weights = nn.Parameter( torch.randn( 1, dtype = torch.float ), requires_grad = True )
            self.bias = nn.Parameter( torch.randn( 1, dtype = torch.float ), requires_grad = True )

        if self._modelVersion == tongue.LINEAR_REGRESSION_VERSION2:
            self.linear_layer = nn.Linear( in_features = 1, out_features = 1 )
    
    def forward( self, x: torch.Tensor ):
        if self._modelVersion == tongue.LINEAR_REGRESSION_VERSION1:
            return self.weights * x + self.bias
        if self._modelVersion == tongue.LINEAR_REGRESSION_VERSION2:
            return self.linear_layer( x )
