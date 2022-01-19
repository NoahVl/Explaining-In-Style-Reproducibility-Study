# Dataloader based on https://github.com/royorel/FFHQ-Aging-Dataset/blob/master/data_loader.py

import torch.utils.data as data
import os
from PIL import Image
import pandas as pd
from torchvision import transforms


class FFHQ(data.Dataset):
    def __init__(self, root, label="gender"):
        """
        PyTorch DataSet for the FFHQ-Age dataset.
        :param root: Root folder that contains a directory for the dataset and the csv with labels in the root directory.
        :param label: Label we want to train on, chosen from the csv labels list.
        """
        self.root = root
        self.target_class = label

        # Store image paths
        self.images = [os.path.join(self.root, "flickrfaceshq-dataset-nvidia-resized-256px", "resized", file)
                       for file in os.listdir(os.path.join(self.root, "flickrfaceshq-dataset-nvidia-resized-256px", "resized")) if file.endswith('.jpg')]

        # Import labels from a CSV file
        self.labels = pd.read_csv(os.path.join(self.root, "ffhq_aging_labels.csv"))

        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize(224),  # Used to be resize 256
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Make a lookup dictionary for the labels
        # Get column names of dataframe
        cols = self.labels.columns.values
        label_ids = {col_name: i for i, col_name in enumerate(cols)}
        self.class_id = label_ids[self.target_class]

        self.one_hot_encoding = {"male": 0,
                                 "female": 1}


    def __getitem__(self, index):
        _img = self.transform(Image.open(self.images[index]))
        _label = self.one_hot_encoding[self.labels.iloc[index, self.class_id]]
        return _img, _label

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    dataset = FFHQ(".")
    print(len(dataset))
    print(dataset[0])

    # Import plt and display the image
    import matplotlib.pyplot as plt
    # Put the image class in the title
    plt.title(dataset[0][1])
    plt.imshow(dataset[0][0].permute(1, 2, 0))
    plt.show()