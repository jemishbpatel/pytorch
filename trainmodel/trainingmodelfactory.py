from utility import tongue
from trainmodel.linearregressiontraining import LinearRegressionTraining
from trainmodel.classficationtraining import BinaryClassficationTraining
from trainmodel.multiclassclassificationtraining import MulticlassClassficationTraining

TRAINING_LOOP_MAP = { tongue.LINEAR_REGRESSION: LinearRegressionTraining,
                        tongue.BINARY_CLASSIFICATION: BinaryClassficationTraining,
                        tongue.MULTILCLASS_CLASSIFICATION : MulticlassClassficationTraining }
def TrainModelFactory( dataType = tongue.LINEAR_REGRESSION ):
    return TRAINING_LOOP_MAP[ dataType ] 
