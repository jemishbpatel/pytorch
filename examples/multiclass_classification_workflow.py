import torch
import logging

logging.basicConfig( filename='multiclass_classification.log', encoding='utf-8', level=logging.INFO )

from utility import tongue
from dataloading.muliclassblobs import MulticlassBlobs
from splittrainandtest.testandtrainsplit import TestAndTrainSplit
from visualize.plot import Visualize
from createmodel.multiclassclassification import MulticlassClassification
from lossfunction.lossfunctionfactory import LossFunctionFactory
from optimizer.optimizerfactory import OptmizerFactory
from trainmodel.trainingmodelfactory import TrainModelFactory
from savemodel.save import SaveModel
from loadmodel.load import LoadModel

class MulticlassClassificiationWorkFlow:
    def __init__( self):
        torch.manual_seed( 42 )
        self.model_0 = None
        self.lossFunction = None
        self.optimizer = None
        self._modelDirectoryPath = "models"
        self._modelFilename = "01_pytorch_workflow_multiclass_classification_model_0.pth"
        self._modelState = None
        self._loadedModel = None
        logging.info( f"Running on: {tongue.TARGET_DEVICE}" )

    def initializeData( self ):
        self.X, self.y = MulticlassBlobs( number_of_classes = 4, number_of_features = 2, random_seeds = 42 ).create()

    def testAndTrainSplit( self ):
        self.X_train, self.y_train, self.X_test, self.y_test = TestAndTrainSplit( self.X, self.y, trainPercentage = 80 ).split()
        self.X_train.to( tongue.TARGET_DEVICE )
        self.y_train.to( tongue.TARGET_DEVICE )
        self.X_test.to( tongue.TARGET_DEVICE )
        self.y_test.to( tongue.TARGET_DEVICE )

    def visualizeInitialData( self ):
        Visualize().plotCircles( self.X, self.y )

    def createModel( self ):
        self.model_0 = MulticlassClassification( input_features = 2, output_features = 4, hidden_units = 8 )
        self.model_0.to( tongue.TARGET_DEVICE )
        logging.debug( list( self.model_0.parameters() ) )
        logging.debug( self.model_0.state_dict() )
        logging.debug( { next( self.model_0.parameters() ).device } )

    def modelPredictions( self ):
        with torch.inference_mode():
            y_preds = self.model_0( self.X_test )
        logging.debug( y_preds )

    def visualizeDecisionBoundary( self ):
        Visualize().plotDecisionBoundaries( self.model_0, self.X_train, self.y_train, self.X_test, self.y_test )

    def getLossFunction( self ):
        lossFunctionForBinaryClassification = LossFunctionFactory( data = tongue.MULTILCLASS_CLASSIFICATION )
        self.lossFunction = lossFunctionForBinaryClassification()

    def getOptimizer( self ):
        optimizerForBinaryClassification = OptmizerFactory( data = tongue.MULTILCLASS_CLASSIFICATION )
        self.optimizer = optimizerForBinaryClassification( params = self.model_0.parameters(), lr = 0.1 )

    def train( self ):
        self.trainModel = TrainModelFactory( dataType = tongue.MULTILCLASS_CLASSIFICATION )()
        self.trainModel.trainingLoop( self.model_0, self.X_train, self.y_train, self.X_test, self.y_test, self.lossFunction, self.optimizer, epochs = 2000 )
        self.trainModel.plotLossCurve()

    def displayParameters( self ):
        logging.info( f"Trained Model parameters: {self.model_0.state_dict()}" )

    def saveModel( self ):
        SaveModel( self.model_0, modeldirectoryPath = self._modelDirectoryPath, modelFilename = self._modelFilename )

    def loadModel( self ):
        self._loadedModel = LoadModel( modelType = tongue.MULTILCLASS_CLASSIFICATION, modelPath = self._modelDirectoryPath + "/" + self._modelFilename ).model()
        logging.info( f"Loaded Trained Model parameters: {self._loadedModel.state_dict()}" )

if __name__ == "__main__":
    obj = MulticlassClassificiationWorkFlow()
    obj.initializeData()
    obj.visualizeInitialData()
    obj.testAndTrainSplit()
    obj.createModel()
    obj.getLossFunction()
    obj.getOptimizer()
    obj.train()
    obj.visualizeDecisionBoundary()
    obj.displayParameters()
    obj.modelPredictions()
    obj.saveModel()
    obj.loadModel()
