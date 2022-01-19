import os

import torchvision.datasets as datasets
import torch
from torch.utils import data
from torchvision.transforms import transforms
from PIL import Image


def convert_to_rgb(image):
    image = image.convert('RGB')
    return image

def mnist_train_valid_test_dataset(download_dir, valid_ratio=0.15):
    """
    Imports the MNIST dataset from the PyTorch hub.
    """
    mnist_train = datasets.MNIST(root=download_dir, train=True, download=True)
    mnist_test = datasets.MNIST(root=download_dir, train=False, download=True)

    # If download_dir/raw_images doesn't exist, create it and save all the images there in RGB format.
    if not os.path.exists(os.path.join(download_dir, 'raw_images')):
        os.makedirs(os.path.join(download_dir, 'raw_images'))
        print("Saving all images to disk in png format... might take a while.")
        for dataset in [mnist_train, mnist_test]:
            img_count = 0
            for img, label in dataset:
                img = img.convert("RGB")
                img.save(os.path.join(download_dir, 'raw_images', f"{img_count}-{label}.png"))
                img_count += 1

    mobile_net_transform = transforms.Compose([
            convert_to_rgb,
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    mnist_train = datasets.MNIST(root=download_dir, train=True, download=True, transform=mobile_net_transform)
    valid_length = int(len(mnist_train) * valid_ratio)
    train_length = len(mnist_train) - valid_length

    train_dataset, val_dataset = data.random_split(mnist_train,
                                                   [train_length, valid_length],
                                                   generator=torch.Generator().manual_seed(42)
                                                   )

    test_dataset = datasets.MNIST(root=download_dir, train=False, download=True, transform=mobile_net_transform)

    return train_dataset, val_dataset, test_dataset
