## References
The FFHQ labels `ffhq_aging_labels.csv` have been downloaded from the [FFHQ-Aging Dataset](https://github.com/royorel/FFHQ-Aging-Dataset) repository.
The DataLoader in `data_loader.py` was also inspired by their dataloader.

Special thanks to Kaggle user [xhlulu](https://www.kaggle.com/xhlulu) for uploading a resized version of the [FFHQ dataset](https://www.kaggle.com/xhlulu/flickrfaceshq-dataset-nvidia-resized-256px) which will be downloaded with the IPython Notebook mentioned below. This saves a lot of time when downloading the images.

## How to download this dataset?
* Create a Kaggle account.
* Generate an API key.
* Run the `download_dataset.ipynb` notebook.
* Install the `opendatasets` package like requested (should already be installed on the included environment).
* Fill in the API attributes the notebook asks for when executing the cells.
* At the end you should have a folder called `resized/` which should contain 70k images of faces.