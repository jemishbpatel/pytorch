class TestAndTrainSplit:
    def __init__( self, X, y, trainPercentage = 80 ):
        self.train = X
        self.test = y
        self.percentage = trainPercentage / 100

    def split( self ):
        trainSplitLength = int( self.percentage * len( self.train ) )
        X_train, y_train = self.train[ :trainSplitLength ], self.test[ :trainSplitLength ]
        X_test, y_test = self.train[ trainSplitLength: ], self.test[ trainSplitLength: ]
        return X_train, y_train, X_test, y_test
