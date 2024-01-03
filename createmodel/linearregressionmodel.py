import torch
from torch import nn

class LinearRegressionModel( nn.Module ):
    def __init__( self ):
        super().__init__()
        self.weights = nn.Parameter( torch.randn( 1, dtype = torch.float ), requires_grad = True )
        self.bias = nn.Parameter( torch.randn( 1, dtype = torch.float ), requires_grad = True )
    
    def forward( self, x: torch.Tensor ):
        return self.weights * x + self.bias
