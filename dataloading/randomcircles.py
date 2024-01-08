import logging
import pandas as pd
from sklearn.datasets import make_circles

class Circles:
    def __init__( self, samples = 1000 ):
        self.samples = samples

    def create( self ):
        self.X, self.y = make_circles( self.samples, noise = 0.03, random_state = 42 )
        self.debugSamples( self.samples )
        return self.X, self.y

    def debugSamples( self, samples ):
        circles = pd.DataFrame( { "X1": self.X[:, 0], "X2": self.X[:, 1], "label": self.y } )
        logging.debug( f"{ circles.head( samples ) }" )
        logging.debug( f"{ circles.label.value_counts() }" )
