# FACT: Explaining in Style - Reproducibility Study

<h1 align="center">
<img src="all_user_studies\user_study_images_old_faces\study_1\class_study_0.gif" alt="GIF of user-study" width="400" height="400"</img>
<br>
ReStylEx
<br>
</h1>

## Requirements
Running this notebook requires a CUDA-enabled graphics card. Installing the environment requires Conda.

## Instructions

1. Create the conda environment from the .yaml file.
2. Open jupyter notebook.
3. Select `model_name` to pick the dataset/model on which to show results. Default is 'plant'.
4. Run the notebook! 

## Verifying results

### Using precomputed attfind values (hdf5 files)
Please check out notebook ...

### Producing the attfind values yourself
Please check out notebook ...

## How to train the models?
The StylEx framework consists of two parts, the "pretrained" classifier and the Encoder+GAN.

If you want to train a StylEx model on a new dataset we suggest you first train a new classifier and then provide it to the `cli.py` file to train the StylEx model on this dataset with the new classifier in evaluation mode. If you use a Resnet/Mobilenet model you should only have to change the classifier_name parameter in the `cli.py` file, or change it as a parameter using `--classifier_name <mobilenet/resnet>` when you call the `cli.py` file.

If you want to use a new classifier architecture you should add support for this in one of the `stylex_train` files.

### Training one of the supported classifiers
Two options, Mobilenet or ResNet, ResNet seemed to give better results on small images upscaled to 244px (?) than Mobilenet.


### Train the StylEx model (Encoder and GAN)
Run the `cli.py` file with your desired parameters. As explained before, you have two options for the `stylex_train` file, the normal `stylex_train` has the old architecture, without softaxing and providing the softmaxed classifications to the discriminator. The `stylex_train_new.py` file does include this. However from our limited testing the first version seemed to work better, your mileage may vary. If you want to use the new script, please change the import in the `cli.py` file by setting the `USE_OLD_ARCHITECTURE` to `False`.

Explain parameters here.


## User study
The files of the user study, which has been discussed in the paper, have been included in this repository in the `/all_user_studies` folder.

## License
MIT.

## Acknowledgements
Our repository is based on the StyleGAN2 training code in PyTorch of the amazing repository of Github user lucidrains, [stylegan2-pytorch](https://github.com/lucidrains/stylegan2-pytorch). To their training code we added the StylEx training code.

The original [TensorFlow notebook](https://github.com/google/explaining-in-style/blob/main/Explaining_in_Style_AttFind.ipynb) of the authors, including the AttFind algorithm from the authors has been translated to PyTorch. It has also been used to run their pretrained age StylEx model to extract experimental results. Both notebooks have been included.
