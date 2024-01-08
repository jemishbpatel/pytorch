import torch
from utility import tongue
from createmodel.linearregressionmodel import LinearRegressionModel
from createmodel.circlesmodel import BinaryClassificationModel


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

    def model( self ):
        self.__getModelObject()
        self._model.load_state_dict(  torch.load( f = self._modelPath ) )
        return self._model
