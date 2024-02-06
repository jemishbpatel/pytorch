import matplotlib.pyplot as plt
from utility.helper_functions import plot_predictions, plot_decision_boundary
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

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
