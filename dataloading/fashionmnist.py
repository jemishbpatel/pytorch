import torch
import logging
from torch import nn

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

class FashionMNISTDataset:
    def __init__( self, dataDirectory = "data" ):
        self._dataDirectory = dataDirectory
        self._train_data = None
        self._test_data = None

    def create( self ):
        self._train_data = datasets.FashionMNIST(
                root = self._dataDirectory,
                train = True,
                download = False,
                transform = ToTensor(),
                target_transform = None
                )

        self._test_data = datasets.FashionMNIST(
                root = self._dataDirectory,
                train = False,
                download = False,
                transform = ToTensor()
                )
        return self._train_data, self._test_data
