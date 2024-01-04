import logging
import torch
from pathlib import Path

class SaveModel:
    def __init__( self, model, modeldirectoryPath = "test", modelFilename = "test.ph" ):
        MODEL_PATH = Path( modeldirectoryPath )
        MODEL_PATH.mkdir( parents = True, exist_ok = True )
        MODEL_NAME = modelFilename
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
        logging.info( f"Saving model to: {MODEL_SAVE_PATH}" )
        torch.save( obj = model.state_dict(), f = MODEL_SAVE_PATH )
