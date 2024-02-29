import torchvision
import torch
import random
from PIL import Image
from pathlib import Path
from timeit import default_timer as timer
import gradio as gr
from utility import tongue


weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
effnetb2_transforms = weights.transforms()
model = torchvision.models.efficientnet_b2( weights = weights )

model.classifier = torch.nn.Sequential(
            torch.nn.Dropout( p = 0.3, inplace = True ),
            torch.nn.Linear( in_features = 1408,
            out_features = 3, # same number of output units as our number of classes
                bias = True )).to( tongue.TARGET_DEVICE )
model.load_state_dict(  torch.load( f = "models/deploy_model.pth" ) )


def predict(img ):
    class_names = [ 'pizza', 'steak', 'sushi' ]

    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = effnetb2_transforms( img ).unsqueeze( 0 )

    # Put model into evaluation mode and turn on inference mode
    model.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax( model( img ), dim = 1 )

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = { class_names[ i ]: float( pred_probs[ 0 ][ i ] ) for i in range( len( class_names ) ) }

    # Calculate the prediction time
    pred_time = round( timer() - start_time, 5 )

    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time


data_path = Path( "data" )
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

# Get a list of all test image filepaths
test_data_paths = list( Path( test_dir ).glob( "*/*.jpg" ) )

# Randomly select a test image path
random_image_path = random.sample( test_data_paths, k = 1 )[ 0 ]

# Open the target image
image = Image.open( random_image_path )
print( f"[INFO] Predicting on image at path: {random_image_path}\n" )

# Predict on the target image and print out the outputs
pred_dict, pred_time = predict( img = image )
print(f"Prediction label and probability dictionary: \n{pred_dict}")
print(f"Prediction time: {pred_time} seconds")
example_list = [[str(filepath)] for filepath in random.sample(test_data_paths, k=3)]
print ( example_list )

# Create title, description and article strings
title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

# Create the Gradio demo
demo = gr.Interface( fn = predict, # mapping function from input to output
                    inputs = gr.Image( type = "pil" ), # what are the inputs?
                    outputs = [ gr.Label( num_top_classes = 3, label = "Predictions" ), # what are the outputs?
                             gr.Number( label = "Prediction time (s)" ) ], # our fn has two outputs, therefore we have two outputs
                    examples = example_list,
                    title = title,
                    description = description,
                    article = article )

# Launch the demo!
demo.launch( debug = False, # print errors locally?
             share = True ) # generate a publically shareable URL?
