from torch.utils.data import DataLoader

class BatchSizeDataLoader:
    def __init__( self, train_data, test_data, batch_size = 32 ):
        self._batchSize = batch_size
        self._trainDataLoader = None
        self._testDataLoader = None
        self._train_data = train_data
        self._test_data = test_data

    def trainDataLoader( self ):
        self._trainDataLoader = DataLoader( self._train_data,
                                            batch_size = self._batchSize,
                                            shuffle = True
                                            )
        return self._trainDataLoader

    def testDataLoader( self ):
        self._testDataLoader = DataLoader( self._test_data,
                                            batch_size = self._batchSize,
                                            shuffle = False 
                                            )
        return self._testDataLoader
