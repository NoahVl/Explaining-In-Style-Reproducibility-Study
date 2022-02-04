import requests
import os
import zipfile
import shutil
import glob

import torchvision
from torchvision.datasets import ImageFolder
import torch

plantvillage_url = 'https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded'

def download_plantvillage_dataset(root='./'):
  """
  Downloads the plant-village dataset and splits the images into sick and healthy
  leaves.

  Args:
    root: Path where the dataset directory is placed.
  Returns:
    None
  """

  # The dataset directory contains a 'healthy' and a 'sick' folder, corresponding
  # to the two classes into which images are classified.

  healthy_path = os.path.join(root, 'plant-village/healthy')
  sick_path = os.path.join(root, 'plant-village/sick')

  # Temporary path where intermediate files are kept.
  tmp_path = os.path.join(root, 'tmp')

  try:
    os.makedirs(healthy_path, exist_ok=True)
    os.makedirs(sick_path, exist_ok=True)
    os.mkdir(tmp_path)
  except OSError:
    # Most likely the paths already exist
    print('Error while creating directories.')


  zip_path = os.path.join(tmp_path, 'dataset.zip')
  
  # Download the dataset
  print("Downloading dataset...")
  r = requests.get(plantvillage_url, allow_redirects=True)
  open(zip_path, 'wb').write(r.content)

  # Unzip the dataset contents
  print("Unzipping dataset...")
  with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(tmp_path)

  os.remove(zip_path)

  id_gen = 0 # For unique image identifier generation 

  # The default directory structure contains fine-grained distinctions between 
  # plant species and disease types. This code reduces images into the 
  # healthy/sick classes.

  for img_dir in glob.glob(tmp_path + '/Plant_leave_diseases_dataset_without_augmentation/*'):
    if 'healthy' in img_dir:
      dst = healthy_path
    else:
      dst = sick_path
    
    for img_path in glob.glob(img_dir  + '/*'):
      shutil.move(img_path, os.path.join(dst, str(id_gen) + '.jpg'))
      id_gen += 1

  # Remove temp directory
  shutil.rmtree(tmp_path)


def get_train_valid_test_dataset(path='./plant-village', image_size=64, train=0.7, valid=0.2,
                                 test=0.1, seed=42):
  """
  Generates a train, validation and test dataset from a given image folder.

  Args:
    path: Root directory of the dataset.
    train: Train fraction.
    valid: Validation fraction.
    test: Test fraction.
    seed: Seed for reproducibility.
  Returns:
    train_set: Training dataset.
    valid_set: Validation dataset.
    test_set: Test dataset.
  """

  transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])



  dataset = ImageFolder(path, transform=transforms)

  train_cnt = round(train * len(dataset))
  valid_cnt = round(valid * len(dataset))
  test_cnt = round(test * len(dataset))

  # Simplest way to make sure the counts add up to len(dataset)

  diff = len(dataset) - (train_cnt + valid_cnt + test_cnt)

  train_cnt += diff

  train_set, valid_set, test_set = torch.utils.data.random_split(
                                  dataset, [train_cnt, valid_cnt, test_cnt],
                                  generator = torch.Generator().manual_seed(seed)
                                  )
  
  return train_set, valid_set, test_set

if __name__ == "__main__":
  download_plantvillage_dataset("./plant_data")