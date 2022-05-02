# [Re] Explaining in Style: Training a GAN to explain a classifier in StyleSpace

This repository is a re-implementation of [Explaining in Style: Training a GAN to explain a classifier in StyleSpace](https://openaccess.thecvf.com/content/ICCV2021/papers/Lang_Explaining_in_Style_Training_a_GAN_To_Explain_a_Classifier_ICCV_2021_paper.pdf) by Lang et al. (2021).

<h1 align="center">
<img src="all_user_studies\user_study_images_old_faces\study_1\class_study_0.gif" alt="GIF of user-study" align="right"  width="200" height="200"</img>
</h1>

[![DOI](https://zenodo.org/badge/442497190.svg)](https://zenodo.org/badge/latestdoi/442497190) [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](/LICENSE) 

## Paper
**[Re] Explaining in Style: Training a GAN to explain a classifier in StyleSpace**  
Noah Van der Vleuten, Tadija Radusinović, Rick Akkerman, Meilina Reksoprodjo

Paper: https://openreview.net/forum?id=SYUxyazQh0Y

**If you use this for research, please cite our paper:**
```bibtex
@inproceedings{vleuten2022re,
  title     = {[Re] Explaining in Style: Training a {GAN} to explain a classifier in StyleSpace},
  author    = {Noah Van der Vleuten and Tadija Radusinovi{\'c} and Rick Akkerman and Meilina Reksoprodjo},
  booktitle = {ML Reproducibility Challenge 2021 (Fall Edition)},
  year      = {2022},
  url       = {https://openreview.net/forum?id=SYUxyazQh0Y}
}
```

## Requirements
Running this notebook requires a CUDA-enabled graphics card. Installing the environment requires Conda.

## Instructions

1. Create the conda environment from the .yml file.
2. Activate the environment.
3. Open jupyter notebook.
4. Open the `stylex/all_results_notebook.ipynb` notebook.
5. Download the model files as described in the notebook.
6. Select `model_to_choose` to pick the dataset/model on which to show results. Default is 'plant'.


## Verifying results

The `all_results_notebook.ipynb` works with pre-calculated latent vectors to generate results and run the experiments. If you want to generate the latent embeddings yourself, make use of the `run_attfind_combined.ipynb` notebook (similarly, select the appropriate `model_to_choose`). Note that you will have to download the datasets if you want to run AttFind (you can make use of the notebooks in the data folder).

**Warning**: The AttFind procedure is quite slow and may take over an hour depending on your hardware.

## How to train the models?
The StylEx framework consists of two parts, the "pretrained" classifier and the Encoder+GAN.

If you want to train a StylEx model on a new dataset we suggest you first train a new classifier and then provide it to the `cli.py` file to train the StylEx model on this dataset with the new classifier in evaluation mode. If you use a Resnet/Mobilenet model you should only have to change the classifier_name parameter in the `cli.py` file, or change it as a parameter using `--classifier_name <mobilenet/resnet>` when you call the `cli.py` file. 

If you want to use a new classifier architecture you should add support for this in one of the `stylex_train.py` files.

### Training one of the supported classifiers
Natively we support the MobileNet V2 and ResNet architecture. Of the two options, ResNet seemed to give much better results on small images upscaled to 224px than MobileNet. The MobileNet classifier [training code](./stylex/train_mobilenet_classifier.py) has been included, however to reiterate, it is advised to train a ResNet classifier when using small images. We have also observed that unfreezing the layers iteratively by editing a Python file is not that preferred.

Therefore we have also created and included a [notebook](./stylex/classifier_training_celeba.ipynb) that was used to train the ResNet-18 CelebA gender classifier, this classifier was then used to be explained by the StyleGAN model trained on the FFHQ dataset as per directions of the original paper. In the notebook it is also possible to train a MobileNet classifier.

## User study
The files of the user study, which has been discussed in the paper, have been included in this repository in the `/all_user_studies` folder.

## License
[MIT](/LICENSE)

## Acknowledgements
Our repository is based on the StyleGAN2 training code in PyTorch of the amazing repository of Github user lucidrains, [stylegan2-pytorch](https://github.com/lucidrains/stylegan2-pytorch). To their training code we added the StylEx training code.

The original [TensorFlow notebook](https://github.com/google/explaining-in-style/blob/main/Explaining_in_Style_AttFind.ipynb) of the authors, including the AttFind algorithm from the authors has been translated to PyTorch. It has also been used to run their pretrained age StylEx model to extract experimental results. Both notebooks have been included.
