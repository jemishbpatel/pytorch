from torchvision import datasets, transforms

class TransformData:
    def __int__( self ):
        self.data_transform = None

    def createTransform( self ):
        self.data_transform = transforms.Compose( [
        # Resize the images to 64x64
        transforms.Resize( size=( 64, 64 ) ),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip( p = 0.5 ), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
        ] )
        return self.data_transform

    def trainAndTestTransform( self ):
        train_transforms = transforms.Compose([
            transforms.Resize(( 64, 64 )),
            transforms.RandomHorizontalFlip( p = 0.5 ),
            transforms.ToTensor()
            ] )

        # Don't augment test data, only reshape
        test_transforms = transforms.Compose([
            transforms.Resize(( 64, 64 )),
            transforms.ToTensor()
        ] )
        return train_transforms, test_transforms

    def trainAndTestTransformWithAugmentation( self ):
        train_transforms = transforms.Compose( [
            transforms.Resize( ( 224, 224 ) ),
            transforms.TrivialAugmentWide( num_magnitude_bins = 31 ), # how intense
            transforms.ToTensor() # use ToTensor() last to get everything between 0 & 1
        ] )

        # Don't need to perform augmentation on the test data
        test_transforms = transforms.Compose( [
            transforms.Resize(( 224, 224 )),
            transforms.ToTensor()
        ] )
        return train_transforms, test_transforms
