import torch
from trainmodel.basictrainingloop import Training

class BinaryClassficationTraining( Training ):
    def _logits( self, X ):
        return self.model( X ).squeeze()

    def _predictions( self, logits ):
        return torch.round( torch.sigmoid( logits ) )

    def _calculateTrainLoss( self ):
        return self.lossFunction( self.y_logits, self.y_train )

    def _calculateTestLoss( self ):
        return self.lossFunction( self.test_logits, self.y_test )

