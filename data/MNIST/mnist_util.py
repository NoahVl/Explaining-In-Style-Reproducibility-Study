import os

import torchvision.datasets as datasets
import torch
from torch.utils import data
from torchvision.transforms import transforms
from PIL import Image


class MNISTOneVersusAll(data.Dataset):
    """
    MNIST dataset for one-versus-all classification.
    """

    def __init__(self, root, target=8, train=True, transform=None, download=False):
        self.root = root
        self.download = download
        self.target = target
        self.dataset = None
        self.train = train
        self.index = 0

        if train:
            self.dataset = datasets.MNIST(root=self.root, train=True, download=self.download,
                                          transform=transform)
        else:
            self.dataset = datasets.MNIST(root=self.root, train=False, download=self.download,
                                          transform=transform)

    def __getitem__(self, index):
        # Get image and label
        if self.index < len(self.dataset):
            img, target = self.dataset[index]
        else:
            img, target = self.dataset[0]
            self.index = 1

        # Check if target is the same as the target we want, this is necessary for the 1 vs many classifier.
        if target == self.target:
            target = 0
        else:
            target = 1

        self.index += 1
        return img, target

    def __len__(self):
        return len(self.dataset)


def convert_to_rgb(image):
    image = image.convert('RGB')
    return image


def mnist_train_valid_test_dataset(download_dir, target=8, valid_ratio=0.15):
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

    mnist_train = MNISTOneVersusAll(root=download_dir, target=target, train=True, transform=mobile_net_transform)
    valid_length = int(len(mnist_train) * valid_ratio)
    train_length = len(mnist_train) - valid_length

    train_dataset, val_dataset = data.random_split(mnist_train,
                                                   [train_length, valid_length],
                                                   generator=torch.Generator().manual_seed(42)
                                                   )

    test_dataset = datasets.MNIST(root=download_dir, train=False, download=True, transform=mobile_net_transform)

    return train_dataset, val_dataset, test_dataset
