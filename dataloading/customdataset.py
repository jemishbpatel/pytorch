import requests
import zipfile
import logging
from pathlib import Path
from torchvision import datasets
from dataloading.imagedataloader import ImageFolderCustom
from transform.transformdata import TransformData

class CustomDataset:
    def __init__( self, dataDirectory = "data" ):
        self.data_path = Path( dataDirectory )
        self.image_path = self.data_path / "pizza_steak_sushi"
        self._train_data = None
        self._test_data = None
        self._train_dir = self.image_path / "train"
        self._test_dir = self.image_path / "test"

    def download( self ):
        if self.image_path.is_dir():
            logging.info( f"{self.image_path} directory exists." )
        else:
            logging.info( f"Did not find {self.image_path} directory, creating one..." )
            self.image_path.mkdir (parents = True, exist_ok = True )
        with open( self.data_path / "pizza_steak_sushi.zip", "wb" ) as f:
            request = requests.get( "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip" )
            logging.info( "Downloading pizza, steak, sushi data..." )
            f.write( request.content )

        with zipfile.ZipFile( self.data_path / "pizza_steak_sushi.zip", "r" ) as zip_ref:
            logging.info( "Unzipping pizza, steak, sushi data..." )
            zip_ref.extractall( self.image_path )

    def imagePath( self ):
        return self.image_path

    def create( self, data_transform ):
        self._train_data = datasets.ImageFolder( root = self._train_dir, transform = data_transform,
                                            target_transform = None )
        self._test_data = datasets.ImageFolder( root = self._test_dir, transform = data_transform )
        return self._train_data, self._test_data

    def createFromCustomImageFolder( self, data_transform ):
        train_transform, test_transforms = TransformData().trainAndTestTransform()
        train_data_custom = ImageFolderCustom( targ_dir = self._train_dir,
                                       transform = train_transform )
        test_data_custom = ImageFolderCustom( targ_dir = self._test_dir,
                                      transform = test_transforms )
        return train_data_custom, test_data_custom

