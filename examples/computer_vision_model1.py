import torch
import logging
import random
from tqdm.auto import tqdm

logging.basicConfig( filename='computer_vision_model1.log', encoding='utf-8', level=logging.INFO )

from utility import tongue
from dataloading.fashionmnist import FashionMNISTDataset
from splittrainandtest.testandtrainsplit import TestAndTrainSplit
from visualize.plot import Visualize
from createmodel.computervisionfashionmnist_cnn import ComputerVisionFashionMNISTModelCNN
from dataloading.dataloader import BatchSizeDataLoader
from lossfunction.lossfunctionfactory import LossFunctionFactory
from optimizer.optimizerfactory import OptmizerFactory
from trainmodel.computervisiontrainloop import ComputerVisionTrainloop
from savemodel.save import SaveModel
from loadmodel.load import LoadModel

class ComputerVisionModel1:
    def __init__( self):
        torch.manual_seed( 42 )
        self.model = None
        self.lossFunction = None
        self.optimizer = None
        self._modelDirectoryPath = "models"
        self._modelFilename = "01_pytorch_workflow_computer_vision_model_cnn.pth"
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
        self.model = ComputerVisionFashionMNISTModelCNN( input_shape = 1, hidden_units = 10, output_shape = len( self.train_data.classes ))
        self.model.to( tongue.TARGET_DEVICE )
        logging.debug( list( self.model.parameters() ) )
        logging.debug( self.model.state_dict() )
        logging.debug( { next( self.model.parameters() ).device } )

    def getDataLoaders( self ):
        batchSizeDataLoader = BatchSizeDataLoader( self.train_data, self.test_data )
        self.train_dataloader = batchSizeDataLoader.trainDataLoader()
        self.test_dataloader = batchSizeDataLoader.testDataLoader()

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
    obj = ComputerVisionModel1()
    obj.initializeData()
    obj.visualizeInitialData()
    obj.createModel()
    obj.getLossFunction()
    obj.getOptimizer()
    obj.getDataLoaders()
    obj.train()
    obj.getRandomSamplesForPredictions()
    obj.makingPredictions()
    obj.saveModel()
    obj.loadModel()
