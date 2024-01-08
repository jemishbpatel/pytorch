from torch import nn
from utility import tongue

class CircleModel( nn.Module ):
    def __init__( self, modelVersion = tongue.CIRCLE_MODEL_VERSION1 ):
        super().__init__()
        self.modelVersion = tongue.CIRCLE_MODEL_VERSION1
        self.layer_1 = nn.Linear( in_features = 2, out_features = 10 )
        self.layer_2 = nn.Linear( in_features = 10, out_features = 10 )
        self.layer_3 = nn.Linear( in_features = 10, out_features = 1 )
        self.relu = nn.ReLU()

    def forward( self, x ):
        return self.layer_3( self.relu( self.layer_2( self.relu(self.layer_1( x ) ) ) ) )
