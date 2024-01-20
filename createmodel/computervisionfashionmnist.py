from torch import nn
from utility import tongue

class ComputerVisionFashionMNISTModelV0( nn.Module ):
    def __init__( self, input_shape: int, hidden_units: int, output_shape: int, model_type = tongue.LINEAR_MODEL_TYPE ):
        super().__init__()
        if tongue.LINEAR_MODEL_TYPE == model_type:
            self.layer_stack = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear( in_features = input_shape, out_features = hidden_units ),
                    nn.Linear( in_features = hidden_units, out_features = output_shape )
                )
        elif tongue.NON_LINEAR_MODEL_TYPE == model_type:
            print( f"in_features : {input_shape} out_features: {output_shape}")
            self.layer_stack = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear( in_features = input_shape, out_features = hidden_units ),
                    nn.ReLU(),
                    nn.Linear( in_features = hidden_units, out_features = output_shape ),
                    nn.ReLU()
            )

    def forward( self, x ):
        return self.layer_stack( x )
