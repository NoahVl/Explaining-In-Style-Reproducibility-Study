import os

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import transforms


def load_classifier(model_name: str, cuda_rank: int, output_size: int = 2) -> torch.nn.Module:
    """
    Returns a MobileNet model with pretrained weights using the model name.
    """

    # Decide which device to put it on (dirty because we should use cuda rank
    device = torch.device(f"cuda:{cuda_rank}") if torch.cuda.is_available() else torch.device("cpu")

    # Load the pretrained mobilenet model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True).to(device)

    # Make the last layer have only output_size outputs instead of 1000.
    model.classifier[1] = nn.Linear(1280, output_size).to(device)

    # Load the weights from the checkpoint.
    model.load_state_dict(torch.load(os.path.join("trained_classifiers", model_name), map_location=device))

    return model


class MobileNet():
    def __init__(self, model_name: str, cuda_rank: int, output_size: int = 2, image_size=32, normalize=True):
        self.model = load_classifier(model_name, cuda_rank, output_size)

        self.mobilenet_dim = 224

        # Image transformation
        self.image_transform = transforms.Compose([
            # I am commenting out resize, since it seems to be automatically done
            # And we need to interpolate to image_size before passing to the classifier
            # transforms.Resize(self.mobilenet_dim), 
            transforms.ToTensor()
        ])

        self.tensor_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Put the model in evaluation mode.
        self.model.eval()

    def classify_images(self, images) -> torch.Tensor:
        """
        Classifies a batch of images using the given model.
        """
        if isinstance(images, torch.Tensor):
            preprocessed_images = F.interpolate(images, size=image_size)
        else:
            preprocessed_images = self.image_transform(images)
            preprocessed_images = F.interpolate(images, size=image_size)

        # I trained on MNIST without normalizing, but it still worked,
        # so I made normalization optional
        if normalize:
            preprocessed_images = self.tensor_transform(preprocessed_images)

        # Classify the images.
        return self.model(images)
