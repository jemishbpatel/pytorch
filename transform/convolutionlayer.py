from torch import nn

class Convolutionlayer:
    def __init__( self, inChannles, outChannels, kernelSize, patchSize, paddingRequired ):

        # Create the Conv2d layer with hyperparameters from the ViT paper
        self.conv2d = nn.Conv2d( in_channels = inChannles, # number of color channels
                    out_channels = outChannels, # from Table 1: Hidden size D, this is the embedding size
                    kernel_size = patchSize, # could also use (patch_size, patch_size)
                    stride = patchSize,
                    padding = paddingRequired )

    def convolution2D( self ):
        return self.conv2d
