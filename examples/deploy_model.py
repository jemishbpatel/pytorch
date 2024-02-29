import torch
import torchvision
import logging
import random
from tqdm.auto import tqdm
from pathlib import Path
logging.basicConfig( filename='deploy_model.log', encoding='utf-8', level=logging.INFO )

from utility import tongue
from utility.helper_functions import plot_loss_curves
from splittrainandtest.testandtrainsplit import TestAndTrainSplit
from visualize.plot import Visualize
from transform.transformdata import TransformData
from dataloading.customdataset import CustomDataset
from dataloading.imagedataloader import ImageFolderCustom
from createmodel.computervisionfashionmnist_cnn import ComputerVisionFashionMNISTModelCNN
from trainmodel.computervisiontrainloop import ComputerVisionTrainloop
from dataloading.dataloader import BatchSizeDataLoader
from lossfunction.lossfunctionfactory import LossFunctionFactory
from optimizer.optimizerfactory import OptmizerFactory
from savemodel.save import SaveModel
from loadmodel.load import LoadModel
from typing import Tuple, Dict

class CustomdatasetModel:
    def __init__( self):
        torch.manual_seed( 42 )
        self.model = None
        self.lossFunction = None
        self.optimizer = None
        self._modelDirectoryPath = "models"
        self._modelFilename = "deploy_model.pth"
        self._modelState = None
        self._loadedModel = None
        logging.info( f"Running on: {tongue.TARGET_DEVICE}" )
        self.train_data = None
        self.test_data = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.image_path = None
        self.weights = None
        self.transformData = TransformData().createTransform()

    def initializeData( self ):
        self.dataset = CustomDataset( "data" )
        self.image_path = self.dataset.imagePath()
        # Get a set of pretrained model weights
        self.weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet
        auto_transforms = self.weights.transforms()
        self.transformData = auto_transforms
        self.train_data, self.test_data = self.dataset.create( self.transformData )
        logging.info( f"Debug classes {self.train_data.classes}" )

    def visualizeInitialData( self ):
        Visualize().randomImage( self.image_path )

    def createModel( self ):
        self.model = torchvision.models.efficientnet_b2( weights = self.weights )
        self.model.to( tongue.TARGET_DEVICE )
        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Recreate the classifier layer and seed it to the target device
        self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout( p = 0.3, inplace = True ), 
                torch.nn.Linear( in_features = 1408, 
                out_features = len( self.train_data.classes ), # same number of output units as our number of classes
                    bias = True )).to( tongue.TARGET_DEVICE )       

    def getDataLoaders( self ):
        batchSizeDataLoader = BatchSizeDataLoader( self.train_data, self.test_data, batch_size = 32 )
        self.train_dataloader = batchSizeDataLoader.trainDataLoader()
        self.test_dataloader = batchSizeDataLoader.testDataLoader()
        img_custom, label_custom = next( iter( self.train_dataloader ))
        logging.info( f"Debug {img_custom.shape} {label_custom.shape}" )

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

    def predict( img ):
        """Transforms and performs a prediction on img and returns prediction and time taken.
        """
        # Start the timer
        start_time = timer()
    
        # Transform the target image and add a batch dimension
        img = effnetb2_transforms(img).unsqueeze(0)
    
        # Put model into evaluation mode and turn on inference mode
        self.model.eval()
        with torch.inference_mode():
            # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
            pred_probs = torch.softmax( self.model( img ), dim = 1 )
    
        # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
        pred_labels_and_probs = { class_names[ i ]: float( pred_probs[ 0 ][ i ] ) for i in range( len( class_names ) ) }
    
        # Calculate the prediction time
        pred_time = round( timer() - start_time, 5 )
    
        # Return the prediction dictionary and prediction time 
        return pred_labels_and_probs, pred_time

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
        lossFunctionForBinaryClassification = LossFunctionFactory( data = tongue.EFFICIENTNET_B2_MODEL )
        self.lossFunction = lossFunctionForBinaryClassification()

    def getOptimizer( self ):
        optimizerForBinaryClassification = OptmizerFactory( data = tongue.EFFICIENTNET_B2_MODEL )
        self.optimizer = optimizerForBinaryClassification( params = self.model.parameters(), lr = 0.001 )

    def train( self ):
        self.trainModel = ComputerVisionTrainloop( self.model, self.train_dataloader, self.test_dataloader, self.lossFunction, self.optimizer, tongue.TARGET_DEVICE, epochs = 10 )
        results = self.trainModel.trainingLoop()
        logging.info( f"results {results}" )
        with torch.no_grad():
            plot_loss_curves( results )

    def saveModel( self ):
        SaveModel( self.model, modeldirectoryPath = self._modelDirectoryPath, modelFilename = self._modelFilename )

    def loadModel( self ):
        self._loadedModel = LoadModel( modelType = tongue.EFFICIENTNET_B2_MODEL, modelPath = self._modelDirectoryPath + "/" + self._modelFilename ).model()
        logging.info( f"Loaded Trained Model parameters: {self._loadedModel.state_dict()}" )

if __name__ == "__main__":
    obj = CustomdatasetModel()
    obj.initializeData()
    obj.getDataLoaders()
    obj.createModel()
    obj.getLossFunction()
    obj.getOptimizer()
    obj.train()
    obj.getRandomSamplesForPredictions()
    obj.saveModel()
