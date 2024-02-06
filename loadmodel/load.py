import torch
from utility import tongue
from createmodel.linearregressionmodel import LinearRegressionModel
from createmodel.binaryclassificationmodel import BinaryClassificationModel
from createmodel.multiclassclassification import MulticlassClassification
from createmodel.computervisionfashionmnist_cnn import ComputerVisionFashionMNISTModelCNN 


class LoadModel:
    def __init__( self, modelType = tongue.LINEAR_REGRESSION,  modelPath = "models" ):
        self._model = None
        self._modelPath = modelPath
        self._modelType = modelType

    def __getModelObject( self ):
        if self._modelType == tongue.LINEAR_REGRESSION:
            self._model =  LinearRegressionModel( modelVersion = tongue.LINEAR_REGRESSION_VERSION2 )
        if self._modelType == tongue.BINARY_CLASSIFICATION:
            self._model =  BinaryClassificationModel( modelVersion = tongue.CIRCLE_MODEL_VERSION1 )
        if self._modelType == tongue.MULTILCLASS_CLASSIFICATION:
            self._model =  MulticlassClassification( input_features = 2, output_features = 4, hidden_units = 8 )
        if self._modelType == tongue.COMPUTER_VISION_MODEL:
            self._model = ComputerVisionFashionMNISTModelCNN( input_shape = 1, hidden_units = 10, output_shape = 10 )

    def model( self ):
        self.__getModelObject()
        self._model.load_state_dict(  torch.load( f = self._modelPath ) )
        return self._model
