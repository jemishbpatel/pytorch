import logging
import torch
from sklearn.datasets import make_blobs

class MulticlassBlobs:
    def __init__( self, number_of_classes = 4, number_of_features = 2, random_seeds = 42 ):
        self._numberOfClasses = number_of_classes
        self._numberOfFeatues = number_of_features
        self._randomSeeds = random_seeds

    def create( self ):
        X_blob, y_blob = make_blobs( n_samples = 1000,
                                        n_features = self._numberOfFeatues,
                                        centers = self._numberOfClasses,
                                        cluster_std = 1.5,
                                        random_state = self._randomSeeds )
        X_blob = torch.from_numpy( X_blob ).type( torch.float )
        y_blob = torch.from_numpy( y_blob ).type( torch.LongTensor )
        logging.debug( X_blob[ :5 ], y_blob[ :5 ] )
        return X_blob, y_blob
