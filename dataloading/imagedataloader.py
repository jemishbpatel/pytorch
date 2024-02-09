# Write a custom dataset class (inherits from torch.utils.data.Dataset)
import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List
from utility.helper_functions import find_classes

# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom( Dataset ):

    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__( self, targ_dir: str, transform=None ):

        # 3. Create class attributes
        # Get all image paths
        self.paths = list( pathlib.Path( targ_dir ).glob( "*/*.jpg" ) ) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes( targ_dir )

    # 4. Make function to load images
    def load_image( self, index: int ):
        "Opens an image via a path and returns it."
        image_path = self.paths[ index ]
        return Image.open( image_path )

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__( self ):
        "Returns the total number of samples."
        return len( self.paths )

    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__( self, index: int ):
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image( index )
        class_name  = self.paths[ index ].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[ class_name ]

        # Transform if necessary
        if self.transform:
            return self.transform( img ), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)
