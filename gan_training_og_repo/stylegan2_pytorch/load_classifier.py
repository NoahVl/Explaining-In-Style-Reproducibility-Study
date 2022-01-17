import os

import torch
from torch import nn


def load_classifier(model_name: str, cuda_rank: int) -> torch.nn.Module:
    """
    Returns a MobileNet model with pretrained weights using the model name.
    """

    # Decide which device to put it on (dirty because we should use cuda rank
    device = torch.device(f"cuda:{cuda_rank}") if torch.cuda.is_available() else torch.device("cpu")

    # Load the pretrained mobilenet model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True).to(device)

    # Make the last layer have only 2 outputs instead of 1000.
    model.classifier[1] = nn.Linear(1280, 2).to(device)

    # Load the weights from the checkpoint.
    model.load_state_dict(torch.load(os.path.join("saved_classifiers", model_name), map_location=device))

    return model
