import torch

# tools used or loading cifar10 dataset
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

from data.Kaggle_FFHQ_Resized_256px.data_loader import FFHQ


def get_train_valid_test_dataset(data_dir, label, valid_ratio=0.15, test_ratio=0.15):
    # TODO: Specify different training routines here per class (such as random crop, random horizontal flip, etc.)

    dataset = FFHQ(data_dir, label)
    train_length, valid_length, test_length = int(len(dataset) * (1 - valid_ratio - test_ratio)), \
                                              int(len(dataset) * valid_ratio), int(len(dataset) * test_ratio)
    # Make sure that the lengths sum to the total length of the dataset
    remainder = len(dataset) - train_length - valid_length - test_length
    train_length += remainder
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                             [train_length, valid_length, test_length],
                                                                             generator=torch.Generator().manual_seed(42)
                                                                             )

    return train_dataset, val_dataset, test_dataset
