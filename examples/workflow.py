import torch
import logging
from utility import tongue
from dataloading.randomdata import RandomDataCreation
from loadmodel.testandtrainsplit import TestAndTrainSplit
from visualize.plot import Visualize
from createmodel.linearregressionmodel import LinearRegressionModel
from lossfunction.lossfunctionfactory import LossFunctionFactory
from optimizer.optimizerfactory import OptmizerFactory 

class Workflow1:
    def __init__( self):
        self.X, self.y = RandomDataCreation( weight = 0.7, bias = 0.3 ).create()
        self.X_train, self.y_train, self.X_test, self.y_test = TestAndTrainSplit( self.X, self.y, trainPercentage = 80 ).split()
        Visualize().plotPredictions( self.X_train, self.y_train, self.X_test, self.y_test, predictions = None )

if __name__ == "__main__":
    obj = Workflow1()
    torch.manual_seed( 42 )
    model_0 = LinearRegressionModel()
    logging.info( list( model_0.parameters() ) )
    logging.info( model_0.state_dict() )
    with torch.inference_mode():
        y_preds = model_0( obj.X_test )
    
    logging.info( y_preds )

    Visualize().plotPredictions( obj.X_train, obj.y_train, obj.X_test, obj.y_test, predictions = y_preds )
    lossFunctionForLinearRegression = LossFunctionFactory( data = tongue.LINEAR_REGRESSION )
    optimizerForLinearRegression = OptmizerFactory( data = tongue.LINEAR_REGRESSION )
    lossFunction = lossFunctionForLinearRegression()
    optimizer = optimizerForLinearRegression( params = model_0.parameters(), lr = 0.01 )
