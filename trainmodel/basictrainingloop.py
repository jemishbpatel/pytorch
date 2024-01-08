import logging
import torch
from tqdm import tqdm
from visualize.plot import Visualize
from utility.dailyhelp import accuracy_fn

class Training:
    def __init__( self ):
        self.train_loss_values = []
        self.test_loss_values = []
        self.epoch_count = []
        self.model = None
        self.y_logits = None
        self.test_logits = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.trainLoss = None
        self.testLoss = None
        self.trainAccuracy = None
        self.testAccuracy = None
        self.lossFunction = None

    def trainingLoop( self, model, X_train, y_train, X_test, y_test, lossFunction, optimizer, epochs = 100 ):
        self._initValues( model, X_train, y_train, X_test, y_test, lossFunction )
        for epoch in tqdm( range( epochs ) ):
            self.model.train()
            self.y_logits = self._logits( self.X_train )
            if self.y_logits == None:
                self.y_logits = self.X_train
            self.y_pred = self._predictions( self.y_logits )
            self.trainLoss = self._calculateTrainLoss()
            self.trainAccuracy = self._calculateAccuracy( self.y_train, self.y_pred )
            optimizer.zero_grad()
            self.trainLoss.backward()
            optimizer.step()
            self.model.eval()
            with torch.inference_mode():
                self.test_logits = self._logits( self.X_test )
                if self.test_logits == None:
                    self.test_logits = self.X_test
                self.test_pred = self._predictions( self.test_logits )
                self.testLoss = self._calculateTestLoss()
                self.testAccuracy = self._calculateAccuracy( self.y_test, self.test_pred )
                if epoch % 10 == 0:
                    self.epoch_count.append( epoch )
                    self.train_loss_values.append( self.trainLoss.detach().numpy() )
                    self.test_loss_values.append( self.testLoss.detach().numpy() )
                    logging.info( f"Epoch: {epoch} | Train Loss: {self.trainLoss} | Test Loss: {self.testLoss} Accuracy: {self.testAccuracy} ")
    def _initValues( self, model, X_train, y_train, X_test, y_test, lossFunction ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.lossFunction = lossFunction

    def plotLossCurve( self ):
        Visualize().plotLossCurve( self.epoch_count, self.train_loss_values, self.test_loss_values )

    def _calculateAccuracy( self, trueValues, predictionData ):
        return accuracy_fn( y_true = trueValues, y_pred = predictionData )

    def _logits( self, unused ):
        return None

    def _predictions( self, X ):
        return self.model( X )

    def _calculateTestLoss( self ):
        pass

    def _calculateTrainLoss( self ):
        pass
