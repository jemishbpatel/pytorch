import torch


class RandomDataCreation:
    def __init__( self, weight = 0.7, bias = 0.3 ):
        self._weight = weight
        self._bias   = bias
    
    def create( self ):
        start = 0
        end = 1
        step = 0.02
        X = torch.arange( start, end, step ).unsqueeze( dim = 1 )
        y = self._weight * X + self._bias
        return X, y

if __name__ == "__main__":
    obj = RandomDataCreation()
    obj.create()
