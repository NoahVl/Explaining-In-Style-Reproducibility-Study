# For the attribute find functions.
from stylex_train import StylEx, Dataset, DistributedSampler, MNIST_1vA

# For setting data src.
from stylex_train import image_noise, styles_def_to_tensor, make_weights_for_balanced_classes, cycle, default

from mobilenet_classifier import MobileNet
import torch
from torch.utils import data
import math

import multiprocessing

NUM_CORES = multiprocessing.cpu_count()


def set_data_src(folder='./', dataset_name=None, image_size=32, batch_size=16, num_workers=4,
                 is_ddp=False, rank=0, world_size=1):
    if dataset_name is None:
        dataset = Dataset(folder, image_size)
        num_workers = default(num_workers, NUM_CORES if not is_ddp else 0)

        sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size,
                                     shuffle=True) if is_ddp else None

        dataloader = data.DataLoader(dataset, num_workers=num_workers,
                                     batch_size=math.ceil(batch_size / world_size), sampler=sampler,
                                     shuffle=not is_ddp, drop_last=True, pin_memory=True)

    elif dataset_name == 'MNIST':

        dataset = MNIST_1vA(digit=8)

        # weights = make_weights_for_balanced_classes(dataset.dataset, num_classes)
        # sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

        dataloader = data.DataLoader(dataset, batch_size=batch_size)
    else:
        raise NotImplementedError("This dataset is not supported yet. Please use dataset_name = None.")

    loader = cycle(dataloader)

    return dataset, loader


def run_attrfind(
        data='./',
        stylex_path='',
        classifier_name='',
        image_size=32,
        n_images=4,
        batch_size=16,
        dataset_name=None,
        s_shift_size=1
):
    dataset, loader = set_data_src(data, dataset_name, image_size, batch_size)
    # Since attribute find is not helped by having multiple GPU's, we hardcode cuda_rank to 0.
    cuda_rank = 0

    stylex = StylEx(image_size=image_size)
    classifier = MobileNet(classifier_name, cuda_rank=cuda_rank, output_size=2, image_size=image_size)

    # Get the latent vectors of the images w.

    # dlatents = []
    # w_styles = []

    if n_images < 0:
        n_images = len(dataset)

    minimums = None
    maximums = None

    for _ in range(n_images // batch_size):
        encoder_batch = next(loader).cuda(cuda_rank)

        encoder_output = stylex.encoder(encoder_batch)
        real_classified_logits = classifier.classify_images(encoder_batch)

        noise = image_noise(batch_size, image_size, device=cuda_rank)

        latent_w = [(torch.cat((encoder_output, real_classified_logits), dim=1),
                     stylex.G.num_layers)]  # Has to be bracketed because expects a noise mix

        # dlatents.append(encoder_output)``
        w_latent_tensor = styles_def_to_tensor(latent_w)

        rgb, style_coords = stylex.G(w_latent_tensor, noise, get_style_coords=True)

        if minimums is None or maximums is None:
            minimums = style_coords
            maximums = style_coords
        else:
            minimums = torch.minimum(minimums, style_coords)
            maximums = torch.maximum(maximums, style_coords)

    minimums = torch.min(minimums, dim=0)[0]
    maximums = torch.max(maximums, dim=0)[0]

    #   pass style vector to generator
    #   classify the result and save it to base_prob
    #   for sindex in range(minimums.shape):
    #       !!! We need some function to get the layer number and index number from the index !!!
    #

    # Get the latent vectors of the images w.

    # dlatents = []
    # w_styles = []

    

    style_coords_list = []
