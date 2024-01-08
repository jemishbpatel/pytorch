import torch
from trainmodel.basictrainingloop import Training

class LinearRegressionTraining( Training ):
    def _predictions( self, X ):
        return self.model( X )

    def _calculateTrainLoss( self ):
        return self.lossFunction( self.y_pred, self.y_train )

    def _calculateTestLoss( self ):
        return self.lossFunction( self.test_pred, self.y_test.type( torch.float ) )
