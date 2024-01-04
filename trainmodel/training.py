import logging
import torch
from tqdm import tqdm
from visualize.plot import Visualize

class TrainModel:
    def __init__( self ):
        self.train_loss_values = []
        self.test_loss_values = []
        self.epoch_count = []

    def trainingLoop( self, model, X_train, y_train, X_test, y_test, lossFunction, optimizer, epochs = 100 ):
        for epoch in tqdm( range( epochs ) ):
            model.train()
            y_pred = model( X_train )
            loss = lossFunction( y_pred, y_train )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                test_pred = model( X_test )
                test_loss = lossFunction( test_pred, y_test.type( torch.float ))
                if epoch % 10 == 0:
                    self.epoch_count.append( epoch )
                    self.train_loss_values.append( loss.detach().numpy() )
                    self.test_loss_values.append( test_loss.detach().numpy() )
                    logging.info( f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

    def plotLossCurve( self ):
        Visualize().plotLossCurve( self.epoch_count, self.train_loss_values, self.test_loss_values )
