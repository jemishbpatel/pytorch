import torch
from torch import nn
import torchvision
import logging
import random
from tqdm.auto import tqdm
from pathlib import Path
logging.basicConfig( filename='paper_replicating.log', encoding='utf-8', level=logging.INFO )

from torchinfo import summary
from utility import tongue
from utility.helper_functions import plot_loss_curves
from splittrainandtest.testandtrainsplit import TestAndTrainSplit
from visualize.plot import Visualize
from transform.transformdata import TransformData
from dataloading.customdataset import CustomDataset
from dataloading.imagedataloader import ImageFolderCustom
from createmodel.patchembedding import PatchEmbedding
from createmodel.multiheadselfattentionblock import MultiheadSelfAttentionBlock
from createmodel.mlpblock import MLPBlock
from createmodel.vit import ViT
from trainmodel.computervisiontrainloop import ComputerVisionTrainloop
from dataloading.dataloader import BatchSizeDataLoader
from lossfunction.lossfunctionfactory import LossFunctionFactory
from optimizer.optimizerfactory import OptmizerFactory
from savemodel.save import SaveModel
from loadmodel.load import LoadModel
from transform.convolutionlayer import Convolutionlayer

class CustomdatasetModel:
    def __init__( self):
        torch.manual_seed( 42 )
        self.model = None
        self.lossFunction = None
        self.optimizer = None
        self._modelDirectoryPath = "models"
        self._modelFilename = "01_pytorch_workflow_custom_dataset_model.pth"
        self._modelState = None
        self._loadedModel = None
        logging.info( f"Running on: {tongue.TARGET_DEVICE}" )
        self.train_data = None
        self.test_data = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.image_path = None
        self.weights = None
        self.img_batch = None
        self.transformData = TransformData().manualTransforms()

    def initializeData( self ):
        self.dataset = CustomDataset( "data" )
        self.image_path = self.dataset.imagePath()
        # Get a set of pretrained model weights
        self.train_data, self.test_data = self.dataset.create( self.transformData )
        logging.info( f"Debug classes {self.train_data.classes}" )

    def createModel( self ):
        self.model = ViT( num_classes = len( self.train_data.classes ) )

    def getConvolutionLayer( self ):
        conv2D = Convolutionlayer( inChannles = 3, outChannels = 768, kernelSize = 16, patchSize = 16 , paddingRequired = 0 )
        self.conv2D = conv2D.convolution2D()

    def flatten2DFeatureMap( self ):
        # Create flatten layer
        flatten = nn.Flatten( start_dim = 2, # flatten feature_map_height (dimension 2)
                      end_dim = 3 ) # flatten feature_map_width (dimension 3)
        return flatten

    def getDataLoaders( self ):
        batchSizeDataLoader = BatchSizeDataLoader( self.train_data, self.test_data, batch_size = 32 )
        self.train_dataloader = batchSizeDataLoader.trainDataLoader()
        self.test_dataloader = batchSizeDataLoader.testDataLoader()
        self.img_batch, label_custom = next( iter( self.train_dataloader ))

    def patchifyAndTokenEmbedding( self ):

        # Create an instance of patch embedding layer
        patchify = PatchEmbedding( in_channels = 3,
                          patch_size = 16,
                          embedding_dim = 768 )

        # Pass a single image through
        logging.info( f"Input image shape: { self.img_batch[ 0 ].unsqueeze( 0 ).shape }" )
        patch_embedded_image = patchify( self.img_batch[ 0 ].unsqueeze( 0 ) ) # add an extra batch dimension on the 0th index, otherwise will error
        logging.info( f"Output patch embedding shape: { patch_embedded_image.shape }" )
        batch_size = patch_embedded_image.shape[ 0 ]
        embedding_dimension = patch_embedded_image.shape[ -1 ]
        class_token = nn.Parameter( torch.ones( batch_size, 1, embedding_dimension ), requires_grad = True )
        patch_embedded_image_with_class_embedding = torch.cat( ( class_token, patch_embedded_image ), dim = 1 )
        logging.info( patch_embedded_image_with_class_embedding )
        logging.info( f"Sequence of patch embeddings with class token prepended shape: { patch_embedded_image_with_class_embedding.shape } -> [ batch_size, number_of_patches, embedding_dimension ]")
        height = width = 224
        patch_size = 16
        # Calculate N (number of patches)
        number_of_patches = int( ( height * width ) / patch_size**2 )

        # Get embedding dimension
        embedding_dimension = patch_embedded_image_with_class_embedding.shape[ 2 ]

        # Create the learnable 1D position embedding
        position_embedding = nn.Parameter( torch.ones( 1,
                                             number_of_patches + 1,
                                             embedding_dimension ),
                                  requires_grad = True ) # make sure it's learnable

        # Show the first 10 sequences and 10 position embedding values and check the shape of the position embedding
        logging.info( position_embedding[ :, :10, :10 ] )
        logging.info( f"Position embeddding shape: { position_embedding.shape } -> [batch_size, number_of_patches, embedding_dimension ]" )
        # Add the position embedding to the patch and class token embedding
        patch_and_position_embedding = patch_embedded_image_with_class_embedding + position_embedding
        logging.info( patch_and_position_embedding )
        logging.info( f"Patch embeddings, class token prepended and positional embeddings added shape: {patch_and_position_embedding.shape} -> [batch_size, number_of_patches, embedding_dimension]" )
        # Create an instance of MSABlock
        multihead_self_attention_block = MultiheadSelfAttentionBlock( embedding_dim = 768, # from Table 1
                                                                        num_heads = 12 ) # from Table 1

        # Pass patch and position image embedding through MSABlock
        patched_image_through_msa_block = multihead_self_attention_block( patch_and_position_embedding )
        print( f"Input shape of MSA block: {patch_and_position_embedding.shape}" )
        print( f"Output shape MSA block: {patched_image_through_msa_block.shape}" )
        # Create an instance of MLPBlock
        mlp_block = MLPBlock( embedding_dim = 768, # from Table 1
                             mlp_size = 3072, # from Table 1
                      dropout = 0.1 ) # from Table 3

        # Pass output of MSABlock through MLPBlock
        patched_image_through_mlp_block = mlp_block( patched_image_through_msa_block )
        print( f"Input shape of MLP block: {patched_image_through_msa_block.shape}" )
        print( f"Output shape MLP block: {patched_image_through_mlp_block.shape}" )        

    def createVitEncoder( self ):

        # Create a random tensor with same shape as a single image
        random_image_tensor = torch.randn( 1, 3, 224, 224 ) # (batch_size, color_channels, height, width)

        # Create an instance of ViT with the number of classes we're working with (pizza, steak, sushi)

        # Pass the random image tensor to our ViT instance
        vit = ViT( num_classes = len( self.train_data.classes ) )
        vit( random_image_tensor )


        # Print a summary of our custom ViT model using torchinfo (uncomment for actual output)
        summary(model=vit,
         input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
         #col_names=["input_size"], # uncomment for smaller output
         col_names=["input_size", "output_size", "num_params", "trainable"],
         col_width=20,
         row_settings=["var_names"]
        )

    def commentedPartForDebugging( self ):
        return
        Visualize().plot_patch_images( img_custom[ 0 ], label_custom[ 0 ] )
        image_out_of_conv = self.conv2D( img_custom[ 0 ] )
        logging.info( f"Debug {image_out_of_conv.shape} {label_custom.shape}" )
        image_out_of_conv_flattened = self.flatten2DFeatureMap()( image_out_of_conv.unsqueeze( 0 ) )
        image_out_of_conv_flattened_reshaped = image_out_of_conv_flattened.permute( 0, 2, 1 )
        single_flattened_feature_map = image_out_of_conv_flattened_reshaped[ :, :, 0 ]
        Visualize().plot_single_feature_map( single_flattened_feature_map )
        logging.info( f"Debug { image_out_of_conv_flattened_reshaped.shape } { label_custom.shape }" )
        Visualize().plot_convolution_layer_features( image_out_of_conv.unsqueeze( 0 ) )

    def makingPredictions( self ):
        y_preds = []
        self.model.eval()
        with torch.inference_mode():
            for X, y in tqdm( self.test_dataloader, desc = "Making predictions" ):
                # Send data and targets to target device
                X, y = X.to( tongue.TARGET_DEVICE ), y.to( tongue.TARGET_DEVICE )
                # Do the forward pass
                y_logit = self.model( X )
                # Turn predictions from logits -> prediction probabilities -> predictions labels
                y_pred = torch.softmax( y_logit, dim = 1 ).argmax( dim = 1 ) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
                # Put predictions on CPU for evaluation
                y_preds.append( y_pred.cpu() )
                # Concatenate list of predictions into a tensor
                y_pred_tensor = torch.cat( y_preds )
        Visualize().plotConfusionMatrix( y_pred_tensor, self.train_data.classes, self.test_data )

    def modelPredictions( self, data ):
        pred_probs = []
        self.model.eval()
        with torch.inference_mode():
            for sample in data:
                sample = torch.unsqueeze( sample, dim = 0 ).to( tongue.TARGET_DEVICE )
                pred_logit = self.model( sample )
                pred_prob = torch.softmax( pred_logit.squeeze(), dim = 0 )
                pred_probs.append( pred_prob.cpu() )
        return torch.stack( pred_probs )

    def getRandomSamplesForPredictions( self ):
        # Get a random list of image paths from test set
        num_images_to_plot = 3
        test_image_path_list = list( Path( self.image_path / "test" ).glob("*/*.jpg")) # get list all image paths from test data 
        test_image_path_sample = random.sample( population = test_image_path_list, # go through all of the test image paths
                                        k = num_images_to_plot) # randomly select 'k' image paths to pred and plot

        # Make predictions on and plot the images
        for image_path in test_image_path_sample:
            Visualize().pred_and_plot_image( model = self.model, 
                            image_path = image_path,
                            class_names = self.train_data.classes,
                             # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                            image_size = (224, 224) )        

    def getLossFunction( self ):
        lossFunctionForBinaryClassification = LossFunctionFactory( data = tongue.COMPUTER_VISION_MODEL )
        self.lossFunction = lossFunctionForBinaryClassification()

    def getOptimizer( self ):
        optimizerForBinaryClassification = OptmizerFactory( data = tongue.TRANSFER_LEARNING )
        self.optimizer = optimizerForBinaryClassification( params = self.model.parameters(), lr = 3e-3, betas = ( 0.9, 0.999 ),
                                                            weight_decay = 0.3 )

    def train( self ):
        self.trainModel = ComputerVisionTrainloop( self.model, self.train_dataloader, self.test_dataloader, self.lossFunction, self.optimizer, tongue.TARGET_DEVICE, epochs = 5 )
        results = self.trainModel.trainingLoop()
        logging.info( f"results {results}" )
        with torch.no_grad():
            plot_loss_curves( results )

    def saveModel( self ):
        SaveModel( self.model, modeldirectoryPath = self._modelDirectoryPath, modelFilename = self._modelFilename )

    def loadModel( self ):
        self._loadedModel = LoadModel( modelType = tongue.COMPUTER_VISION_MODEL, modelPath = self._modelDirectoryPath + "/" + self._modelFilename ).model()
        logging.info( f"Loaded Trained Model parameters: {self._loadedModel.state_dict()}" )

if __name__ == "__main__":
    obj = CustomdatasetModel()
    obj.initializeData()
    obj.getConvolutionLayer()
    obj.getDataLoaders()
    obj.patchifyAndTokenEmbedding()
    obj.createVitEncoder()
    obj.createModel()
    obj.getLossFunction()
    obj.getOptimizer()
    obj.train()
#    obj.getRandomSamplesForPredictions()
