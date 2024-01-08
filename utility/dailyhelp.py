import torch

def accuracy_fn( y_true, y_pred ):
    correct = torch.eq( y_true, y_pred ).sum().item()
    accuracy = ( correct / len( y_pred ) ) * 100
    return accuracy
