import torch
import logging

logging.basicConfig( filename='workflow1.log', encoding='utf-8', level=logging.INFO )

from utility import tongue
from dataloading.randomdata import RandomDataCreation
from splittrainandtest.testandtrainsplit import TestAndTrainSplit
from visualize.plot import Visualize
from createmodel.linearregressionmodel import LinearRegressionModel
from lossfunction.lossfunctionfactory import LossFunctionFactory
from optimizer.optimizerfactory import OptmizerFactory
from trainmodel.training import TrainModel
from savemodel.save import SaveModel
from loadmodel.load import LoadModel

class Workflow1:
    def __init__( self):
        torch.manual_seed( 42 )
        self.model_0 = None
        self.lossFunction = None
        self.optimizer = None
        self.trainModel = TrainModel()
        self._modelDirectoryPath = "models"
        self._modelFilename = "01_pytorch_workflow_model_0.pth"
        self._modelState = None
        self._loadedModel = None
        logging.info( f"Running on: {tongue.TARGET_DEVICE}" )

    def initializeData( self ):
        self.X, self.y = RandomDataCreation( weight = 0.7, bias = 0.3 ).create()

    def testAndTrainSplit( self ):
        self.X_train, self.y_train, self.X_test, self.y_test = TestAndTrainSplit( self.X, self.y, trainPercentage = 80 ).split()
        self.X_train.to( tongue.TARGET_DEVICE )
        self.y_train.to( tongue.TARGET_DEVICE )
        self.X_test.to( tongue.TARGET_DEVICE )
        self.y_test.to( tongue.TARGET_DEVICE )

    def visualizeInitialData( self ):
        Visualize().plotPredictions( self.X_train, self.y_train, self.X_test, self.y_test, predictions = None )

    def createModel( self ):
        self.model_0 = LinearRegressionModel( modelVersion = tongue.LINEAR_REGRESSION_VERSION2 )
        self.model_0.to( tongue.TARGET_DEVICE )
        logging.debug( list( self.model_0.parameters() ) )
        logging.debug( self.model_0.state_dict() )
        logging.debug( { next( self.model_0.parameters() ).device } )

    def modelPredictions( self ):
        with torch.inference_mode():
            y_preds = self.model_0( self.X_test )
        logging.debug( y_preds )
        Visualize().plotPredictions( self.X_train, self.y_train, self.X_test, self.y_test, predictions = y_preds )

    def getLossFunction( self ):
        lossFunctionForLinearRegression = LossFunctionFactory( data = tongue.LINEAR_REGRESSION )
        self.lossFunction = lossFunctionForLinearRegression()

    def getOptimizer( self ):
        optimizerForLinearRegression = OptmizerFactory( data = tongue.LINEAR_REGRESSION )
        self.optimizer = optimizerForLinearRegression( params = self.model_0.parameters(), lr = 0.01 )

    def train( self ):
        self.trainModel.trainingLoop( self.model_0, self.X_train, self.y_train, self.X_test, self.y_test, self.lossFunction, self.optimizer, epochs = 100 )
        self.trainModel.plotLossCurve()

    def displayParameters( self ):
        logging.info( f"Trained Model parameters: {self.model_0.state_dict()}" )

    def saveModel( self ):
        SaveModel( self.model_0, modeldirectoryPath = self._modelDirectoryPath, modelFilename = self._modelFilename )

    def loadModel( self ):
        self._loadedModel = LoadModel( modelType = tongue.LINEAR_REGRESSION, modelPath = self._modelDirectoryPath + "/" + self._modelFilename ).model()
        logging.info( f"Loaded Trained Model parameters: {self._loadedModel.state_dict()}" )

if __name__ == "__main__":
    obj = Workflow1()
    obj.initializeData()
    obj.testAndTrainSplit()
    obj.visualizeInitialData()
    obj.createModel()
    obj.modelPredictions()
    obj.getLossFunction()
    obj.getOptimizer()
    obj.train()
    obj.displayParameters()
    obj.modelPredictions()
    obj.saveModel()
    obj.loadModel()
