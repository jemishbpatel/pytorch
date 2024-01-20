import matplotlib.pyplot as plt
from utility.helper_functions import plot_predictions, plot_decision_boundary

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
