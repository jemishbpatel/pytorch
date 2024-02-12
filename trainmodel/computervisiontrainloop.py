import logging
import torch
from tqdm import tqdm
from timeit import default_timer as timer
from utility.helper_functions import accuracy_fn, print_train_time

class ComputerVisionTrainloop:
    torch.manual_seed( 42 )

    def __init__( self, model, train_dataloader, test_dataloader, lossFunction, optimizer, device, epochs = 3 ):
        self._epochs = epochs
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self._model = model
        self._lossFunction = lossFunction
        self._optimizer = optimizer
        self._device = device

    def trainingLoop( self ):
        results = {
                    "train_loss": [],
                    "train_acc": [],
                    "test_loss": [],
                    "test_acc": []
                }
        train_time_start_on_cpu = timer()
        for epoch in tqdm( range( self._epochs ) ):
            train_loss = 0
            train_acc = 0
            for batch, ( X, y ) in enumerate( self._train_dataloader ):
                X, y = X.to( self._device ), y.to( self._device )
                self._model.train()
                y_pred = self._model( X )

                loss = self._lossFunction( y_pred, y )
                train_loss += loss
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                y_pred_class = torch.argmax( torch.softmax( y_pred, dim = 1 ), dim = 1 )
                train_acc += ( y_pred_class == y ).sum().item() / len( y_pred )
                if batch % 400 == 0:
                     logging.info( f"Looked at { batch * len( X ) }/{ len(self._train_dataloader.dataset ) } samples" )
            train_loss /= len( self._train_dataloader )
            test_loss, test_acc = 0, 0
            self._model.eval()
            with torch.inference_mode():
                for X, y in self._test_dataloader:
                    X, y = X.to( self._device ), y.to( self._device )
                    test_pred = self._model( X )
                    test_loss += self._lossFunction( test_pred, y )
                    test_acc += accuracy_fn( y_true = y, y_pred = test_pred.argmax( dim = 1 ))
                test_loss /= len( self._test_dataloader )
                test_acc /= len( self._test_dataloader )
            results[ "train_loss" ].append( train_loss )
            results[ "train_acc" ].append( train_acc )
            results[ "test_loss" ].append( test_loss )
            results[ "test_acc" ].append( test_acc )

            logging.info( f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n" )
        train_time_end_on_cpu = timer()
        total_train_time_model_0 = print_train_time( start = train_time_start_on_cpu,
                                           end = train_time_end_on_cpu,
                                           device = str( next (self._model.parameters()).device ) )
        return results
