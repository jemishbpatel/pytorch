import torch
import logging
import random
from tqdm.auto import tqdm

logging.basicConfig( filename='custom_dataset_model.log', encoding='utf-8', level=logging.INFO )

from utility import tongue
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
        self.transformData = TransformData().createTransform()
#        self.transformData, self.testtransform = TransformData().trainAndTestTransformWithAugmentation()

    def initializeData( self ):
        self.dataset = CustomDataset( "data" )
        self.image_path = self.dataset.imagePath()
#        self.train_data, self.test_data = self.dataset.createFromCustomImageFolder( self.transformData )
#        self.dataset.download()
        self.train_data, self.test_data = self.dataset.create( self.transformData )
        logging.info( f"Debug classes {self.train_data.classes}" )
#        logging.info( f"Debug train data: {len(self.train_data)} test data: {len(self.test_data)}" )

    def visualizeInitialData( self ):
        Visualize().randomImage( self.image_path )
#        Visualize().plot_transformed_images( list( self.image_path.glob( "*/*/*.jpg" )), self.transformData, n = 3, seed = None )
#        Visualize().display_random_images( self.train_data, self.train_data.classes, n = 5 )

    def createModel( self ):
        self.model = ComputerVisionFashionMNISTModelCNN( input_shape = 3, hidden_units = 10, output_shape = len( self.train_data.classes ))
        self.model.to( tongue.TARGET_DEVICE )
        logging.debug( list( self.model.parameters() ) )
        logging.debug( self.model.state_dict() )
        logging.debug( { next( self.model.parameters() ).device } )

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

    def getRandomSamplesForPredictions( self ):
        test_samples = []
        test_labels = []
        for sample, label in random.sample( list( self.test_data ), k = 9 ):
            test_samples.append( sample )
            test_labels.append( label )
        logging.info(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({self.train_data.classes[test_labels[0]]})")
        pred_probs = self.modelPredictions( data = test_samples )
        logging.info( f"{pred_probs}" )
        pred_classes = pred_probs.argmax( dim = 1 )
        logging.info( f"Pred classes : {pred_classes}" )
        logging.info( f"Test lables : {test_labels}" )
        Visualize().plotComputerVisionPredictions( test_samples, self.train_data.classes, pred_classes, test_labels  )

    def getLossFunction( self ):
        lossFunctionForBinaryClassification = LossFunctionFactory( data = tongue.COMPUTER_VISION_MODEL )
        self.lossFunction = lossFunctionForBinaryClassification()

    def getOptimizer( self ):
        optimizerForBinaryClassification = OptmizerFactory( data = tongue.COMPUTER_VISION_MODEL )
        self.optimizer = optimizerForBinaryClassification( params = self.model.parameters(), lr = 0.1 )

    def train( self ):
        self.trainModel = ComputerVisionTrainloop( self.model, self.train_dataloader, self.test_dataloader, self.lossFunction, self.optimizer, tongue.TARGET_DEVICE, epochs = 3 )
        self.trainModel.trainingLoop()

    def saveModel( self ):
        SaveModel( self.model, modeldirectoryPath = self._modelDirectoryPath, modelFilename = self._modelFilename )

    def loadModel( self ):
        self._loadedModel = LoadModel( modelType = tongue.COMPUTER_VISION_MODEL, modelPath = self._modelDirectoryPath + "/" + self._modelFilename ).model()
        logging.info( f"Loaded Trained Model parameters: {self._loadedModel.state_dict()}" )

if __name__ == "__main__":
    obj = CustomdatasetModel()
    obj.initializeData()
    obj.visualizeInitialData()
    obj.getDataLoaders()
    obj.createModel()
    obj.getLossFunction()
    obj.getOptimizer()
    obj.train()
    obj.getRandomSamplesForPredictions()
    obj.makingPredictions()
#    obj.saveModel()
#    obj.loadModel()
