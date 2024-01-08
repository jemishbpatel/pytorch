from utility import tongue
from trainmodel.linearregressiontraining import LinearRegressionTraining
from trainmodel.classficationtraining import BinaryClassficationTraining

TRAINING_LOOP_MAP = { tongue.LINEAR_REGRESSION: LinearRegressionTraining,
                        tongue.BINARY_CLASSIFICATION: BinaryClassficationTraining }
def TrainModelFactory( dataType = tongue.LINEAR_REGRESSION ):
    return TRAINING_LOOP_MAP[ dataType ] 
