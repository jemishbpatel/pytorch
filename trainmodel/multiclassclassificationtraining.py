import torch
from trainmodel.basictrainingloop import Training

class MulticlassClassficationTraining( Training ):
    def _logits( self, X ):
        return self.model( X )

    def _predictions( self, logits ):
        return torch.softmax( logits, dim = 1  ).argmax( dim = 1 ) 

    def _calculateTrainLoss( self ):
        return self.lossFunction( self.y_logits, self.y_train )

    def _calculateTestLoss( self ):
        return self.lossFunction( self.test_logits, self.y_test )
