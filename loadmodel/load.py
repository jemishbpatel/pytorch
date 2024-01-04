import torch
from utility import tongue
from createmodel.linearregressionmodel import LinearRegressionModel


class LoadModel:
    def __init__( self, modelType = tongue.LINEAR_REGRESSION,  modelPath = "models" ):
        self._model = None
        self._modelPath = modelPath
        self._modelType = modelType

    def __getModelObject( self ):
        if self._modelType == tongue.LINEAR_REGRESSION:
            self._model =  LinearRegressionModel( modelVersion = tongue.LINEAR_REGRESSION_VERSION2 )

    def model( self ):
        self.__getModelObject()
        self._model.load_state_dict(  torch.load( f = self._modelPath ) )
        return self._model
