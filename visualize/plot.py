from torchvision import transforms
import torch
import logging
import random
from PIL import Image
import matplotlib.pyplot as plt
from utility import tongue
from utility.helper_functions import plot_predictions, plot_decision_boundary, plot_loss_curves
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from typing import List, Tuple

class Visualize:
    def plotPredictions( self, trainData = None, trainLables = None, testData = None, testLables = None, predictions = None ):
        assert trainData is not None
        assert trainLables is not None
        assert testData is not None
        assert testLables is not None
        plt.figure( figsize = ( 10, 7 ) )
        plt.scatter(trainData, trainLables, c = "b", s = 4, label = "Training data" )
        plt.scatter(testData, testLables, c = "g", s = 4, label = "Testing data")
        if predictions is not None:
            plt.scatter( testData, predictions, c = "r", s = 4, label = "Predictions" )
        plt.legend( prop = { "size": 14 } )
        plt.show()

    def plotLossCurve( self, epoch_count, train_loss_values, test_loss_values ):
        plt.plot( epoch_count, train_loss_values, label="Train loss" )
        plt.plot( epoch_count, test_loss_values, label="Test loss" )
        plt.title( "Training and test loss curves" )
        plt.ylabel( "Loss" )
        plt.xlabel( "Epochs" )
        plt.legend()
        plt.show()

    def plotCircles( self, X, y ):
        plt.scatter( x = X[:, 0], y = X[:, 1], c = y, cmap = plt.cm.RdYlBu )
        plt.show()

    def plotDecisionBoundaries( self, model, X_train, y_train, X_test, y_test ):
        plt.figure( figsize = ( 12, 6 ) )
        plt.subplot( 1, 2, 1 )
        plt.title( "Train" )
        plot_decision_boundary( model, X_train, y_train )
        plt.subplot( 1, 2, 2 )
        plt.title( "Test" )
        plot_decision_boundary( model, X_test, y_test)
        plt.show()

    def plotSampleImage( self, train_data_sample ):
        image, label = train_data_sample
        plt.imshow( image.squeeze() )
        plt.title( label )
        plt.show()

    def plotComputerVisionPredictions( self, test_samples, class_names, pred_classes, test_labels  ):
        plt.figure( figsize = ( 9, 9 ) )
        nrows = 3
        ncols = 3
        for i, sample in enumerate( test_samples ):
            # Create a subplot
            plt.subplot( nrows, ncols, i + 1 )
            # Plot the target image
            logging.info( f"Debug sample shape : {sample.shape}" )
            plt.imshow( sample.squeeze(), cmap = "gray" )
            # Find the prediction label (in text form, e.g. "Sandal")
            pred_label = class_names[ pred_classes[ i ] ]
            # Get the truth label (in text form, e.g. "T-shirt")
            truth_label = class_names[ test_labels[ i ] ]
            # Create the title text of the plot
            title_text = f"Pred: {pred_label} | Truth: {truth_label}"
            # Check for equality and change title colour accordingly
            if pred_label == truth_label:
                plt.title( title_text, fontsize = 10, c = "g" ) # green text if correct
            else:
                plt.title( title_text, fontsize = 10, c = "r" ) # red text if wrong
            plt.axis(False);
        plt.show()

    def plotConfusionMatrix( self, y_pred_tensor, class_names, test_data ):
        # 2. Setup confusion matrix instance and compare predictions to targets
        confmat = ConfusionMatrix( num_classes = len( class_names ), task = 'multiclass' )
        confmat_tensor = confmat( preds= y_pred_tensor,
                          target = test_data.targets )

        # 3. Plot the confusion matrix
        fig, ax = plot_confusion_matrix(
            conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy
            class_names = class_names, # turn the row and column labels into class names
            figsize = ( 10, 7 )
        );
        plt.show()

    def randomImage( self, image_path ):
        image_path_list = list( image_path.glob( "*/*/*.jpg" ))
        random_image_path = random.choice( image_path_list )
        image_class = random_image_path.parent.stem
        img = Image.open(random_image_path)
        plt.imshow( img )
        plt.show()

    def plot_transformed_images( self, image_paths, transform, n = 3, seed = 42 ):
        random.seed( seed )
        random_image_paths = random.sample( image_paths, k = n )
        for image_path in random_image_paths:
            with Image.open(image_path) as f:
                fig, ax = plt.subplots( 1, 2 )
                ax[ 0 ].imshow( f )
                ax[ 0 ].set_title( f"Original \nSize: {f.size}" )
                ax[ 0 ].axis( "off" )
                transformed_image = transform( f ).permute( 1, 2, 0 )
                ax[ 1 ].imshow( transformed_image )
                ax[ 1 ].set_title( f"Transformed \nSize: {transformed_image.shape}" )
                ax[ 1 ].axis( "off" )
                fig.suptitle( f"Class: {image_path.parent.stem}", fontsize = 16 )
        plt.show()

    # 1. Take in a Dataset as well as a list of class names
    def display_random_images( self, dataset,
                          classes,
                          n = 5,
                          display_shape = True,
                          seed = None ):

        # 2. Adjust display if n too high
        if n > 10:
            n = 10
            display_shape = False
            print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")

        # 3. Set random seed
        if seed:
            random.seed(seed)

        # 4. Get random sample indexes
        random_samples_idx = random.sample( range( len( dataset ) ), k = n )

        # 5. Setup plot
        plt.figure( figsize = ( 16, 8 ) )

        # 6. Loop through samples and display random samples
        for i, targ_sample in enumerate( random_samples_idx ):
            targ_image, targ_label = dataset[ targ_sample ][ 0 ], dataset[ targ_sample ][ 1 ]

            # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
            targ_image_adjust = targ_image.permute( 1, 2, 0 )

            # Plot adjusted samples
            plt.subplot( 1, n, i+1 )
            plt.imshow( targ_image_adjust )
            plt.axis( "off" )
            if classes:
                title = f"class: {classes[targ_label]}"
                if display_shape:
                    title = title + f"\nshape: {targ_image_adjust.shape}"
            plt.title( title )
        plt.show()

    def pred_and_plot_image( self, model,
                        image_path,
                        class_names,
                        image_size = ( 224, 224 ),
                        transform = None,
                        device = tongue.TARGET_DEVICE ):


        # 2. Open image
        img = Image.open( image_path )

        # 3. Create transformation for image (if one doesn't exist)
        if transform is not None:
            image_transform = transform
        else:
            image_transform = transforms.Compose([
                transforms.Resize( image_size ),
                transforms.ToTensor(),
                transforms.Normalize( mean = [0.485, 0.456, 0.406],
                                  std = [0.229, 0.224, 0.225] ),
            ])

        ### Predict on image ###

        # 4. Make sure the model is on the target device
        model.to( device )

        # 5. Turn on model evaluation mode and inference mode
        model.eval()
        with torch.inference_mode():
            # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
            transformed_image = image_transform( img ).unsqueeze( dim = 0 )

            # 7. Make a prediction on image with an extra dimension and send it to the target device
            target_image_pred = model( transformed_image.to( device ) )

        # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
        target_image_pred_probs = torch.softmax( target_image_pred, dim = 1 )

        # 9. Convert prediction probabilities -> prediction labels
        target_image_pred_label = torch.argmax( target_image_pred_probs, dim = 1 )

        # 10. Plot image with predicted label and probability
        plt.figure()
        plt.imshow( img )
        plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
        plt.axis( False )
        plt.show()
