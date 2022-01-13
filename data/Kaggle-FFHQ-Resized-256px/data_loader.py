# Dataloader based on https://github.com/royorel/FFHQ-Aging-Dataset/blob/master/data_loader.py

import torch.utils.data as data
import os
from PIL import Image
import pandas as pd


class FFHQ(data.Dataset):
    def __init__(self, root, label="gender"):
        self.root = root
        self.target_class = label

        # Store image paths
        self.images = [os.path.join(self.root, "flickrfaceshq-dataset-nvidia-resized-256px", "resized", file)
                       for file in os.listdir(os.path.join(self.root, "flickrfaceshq-dataset-nvidia-resized-256px", "resized")) if file.endswith('.jpg')]

        # Import labels from a CSV file
        self.labels = pd.read_csv(os.path.join(self.root, "ffhq_aging_labels.csv"))

        # Make a lookup dictionary for the labels
        # Get column names of dataframe
        cols = self.labels.columns.values
        label_ids = {col_name: i for i, col_name in enumerate(cols)}
        self.class_id = label_ids[self.target_class]

        self.one_hot_encoding = {"male": 0,
                                 "female": 1}


    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
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
    plt.imshow(dataset[0][0])
    plt.show()