import torch
import logging

logging.basicConfig( filename='computer_vision_model1.log', encoding='utf-8', level=logging.INFO )

from utility import tongue
from dataloading.fashionmnist import FashionMNISTDataset
from splittrainandtest.testandtrainsplit import TestAndTrainSplit
from visualize.plot import Visualize
from createmodel.computervisionfashionmnist import ComputerVisionFashionMNISTModelV0
from dataloading.dataloader import BatchSizeDataLoader
from lossfunction.lossfunctionfactory import LossFunctionFactory
from optimizer.optimizerfactory import OptmizerFactory
from trainmodel.computervisiontrainloop import ComputerVisionTrainloop
from savemodel.save import SaveModel
from loadmodel.load import LoadModel

class ComputerVisionModel1:
    def __init__( self):
        torch.manual_seed( 42 )
        self.model_0 = None
        self.lossFunction = None
        self.optimizer = None
        self._modelDirectoryPath = "models"
        self._modelFilename = "01_pytorch_workflow_computer_vision_model_0.pth"
        self._modelState = None
        self._loadedModel = None
        logging.info( f"Running on: {tongue.TARGET_DEVICE}" )
        self.train_data = None
        self.test_data = None
        self.train_dataloader = None
        self.test_dataloader = None

    def initializeData( self ):
        self.train_data, self.test_data = FashionMNISTDataset( dataDirectory = "data" ).create()
        self.X_train = self.train_data.data
        self.y_train = self.train_data.targets
        self.X_test = self.test_data.data
        self.y_test = self.test_data.targets

        self.X_train.to( tongue.TARGET_DEVICE )
        self.y_train.to( tongue.TARGET_DEVICE )
        self.X_test.to( tongue.TARGET_DEVICE )
        self.y_test.to( tongue.TARGET_DEVICE )

    def visualizeInitialData( self ):
        Visualize().plotSampleImage( self.train_data[ 0 ] )

    def createModel( self ):
        self.model_0 = ComputerVisionFashionMNISTModelV0( input_shape = 784, hidden_units = 10, output_shape = len( self.train_data.classes ) , model_type = tongue.NON_LINEAR_MODEL_TYPE )
        self.model_0.to( tongue.TARGET_DEVICE )
        logging.debug( list( self.model_0.parameters() ) )
        logging.debug( self.model_0.state_dict() )
        logging.debug( { next( self.model_0.parameters() ).device } )

    def getDataLoaders( self ):
        batchSizeDataLoader = BatchSizeDataLoader( self.train_data, self.test_data )
        self.train_dataloader = batchSizeDataLoader.trainDataLoader()
        self.test_dataloader = batchSizeDataLoader.testDataLoader()

    def modelPredictions( self ):
        with torch.inference_mode():
            y_preds = self.model_0( self.X_test )
        logging.debug( y_preds )

    def visualizeDecisionBoundary( self ):
        Visualize().plotDecisionBoundaries( self.model_0, self.X_train, self.y_train, self.X_test, self.y_test )

    def getLossFunction( self ):
        lossFunctionForBinaryClassification = LossFunctionFactory( data = tongue.COMPUTER_VISION_MODEL )
        self.lossFunction = lossFunctionForBinaryClassification()

    def getOptimizer( self ):
        optimizerForBinaryClassification = OptmizerFactory( data = tongue.COMPUTER_VISION_MODEL )
        self.optimizer = optimizerForBinaryClassification( params = self.model_0.parameters(), lr = 0.1 )

    def train( self ):
        self.trainModel = ComputerVisionTrainloop( self.model_0, self.train_dataloader, self.test_dataloader, self.lossFunction, self.optimizer, tongue.TARGET_DEVICE, epochs = 3 )
        self.trainModel.trainingLoop()
    def displayParameters( self ):
        logging.info( f"Trained Model parameters: {self.model_0.state_dict()}" )

    def saveModel( self ):
        SaveModel( self.model_0, modeldirectoryPath = self._modelDirectoryPath, modelFilename = self._modelFilename )

    def loadModel( self ):
        self._loadedModel = LoadModel( modelType = tongue.MULTILCLASS_CLASSIFICATION, modelPath = self._modelDirectoryPath + "/" + self._modelFilename ).model()
        logging.info( f"Loaded Trained Model parameters: {self._loadedModel.state_dict()}" )

if __name__ == "__main__":
    obj = ComputerVisionModel1()
    obj.initializeData()
    obj.visualizeInitialData()
    obj.createModel()
    obj.getLossFunction()
    obj.getOptimizer()
    obj.getDataLoaders()
    obj.train()
#    obj.visualizeDecisionBoundary()
#    obj.displayParameters()
#    obj.modelPredictions()
#    obj.saveModel()
#    obj.loadModel()
