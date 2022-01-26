import os
import fire
import random
from retry.api import retry_call
from tqdm import tqdm
from datetime import datetime
from functools import wraps
from stylex_train import Trainer, NanException

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np



def cast_list(el):
    return el if isinstance(el, list) else [el]


def timestamped_filename(prefix='generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def run_training(rank, world_size, model_args, data, load_from, new, num_train_steps, name, seed, dataset_name=None):
    is_main = rank == 0
    is_ddp = world_size > 1

    if is_ddp:
        set_seed(seed)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        print(f"{rank + 1}/{world_size} process initialized.")

    model_args.update(
        is_ddp=is_ddp,
        rank=rank,
        world_size=world_size
    )

    model = Trainer(**model_args)

    if not new:
        model.load(load_from)
    else:
        model.clear()

    model.set_data_src(data, dataset_name=dataset_name)

    progress_bar = tqdm(initial=model.steps, total=num_train_steps, mininterval=10., desc=f'{name}<{data}>')
    while model.steps < num_train_steps:
        retry_call(model.train, tries=3, exceptions=NanException)
        progress_bar.n = model.steps
        progress_bar.refresh()
        if is_main and model.steps % 50 == 0:
            model.print_log()

    model.save(model.checkpoint_num)

    if is_ddp:
        dist.destroy_process_group()


def train_from_folder(
        data='./mnist_images',  # Used to be data TODO: change back
        results_dir='./results',
        models_dir='./models',
<<<<<<< HEAD:stylex/cli.py
        name='StylEx',  # Used to be 'default' (FFHQ) on my pc TODO: change back
=======
        name='mnist_4_proper_kl',  # Used to be 'default' (FFHQ) on my pc TODO: change back
>>>>>>> main:gan_training_og_repo/stylegan2_pytorch/cli.py
        new=False,
        load_from=-1,
        image_size=32,
        network_capacity=16,
        fmap_max=512,   # 512
        transparent=False,
        batch_size=5,
        gradient_accumulate_every=2,
        num_train_steps=150000,
        learning_rate=2e-4,
        lr_mlp=0.1,
        ttur_mult=1.5,
        rel_disc_loss=False,
        num_workers=4,  # None
        save_every=1000,  # 1000
        evaluate_every=50,  # 1000
        generate=False,
        num_generate=1,
        generate_interpolation=False,
        interpolation_num_steps=100,
        save_frames=False,
        num_image_tiles=8,
        trunc_psi=0.75,
        mixed_prob=0.9,
        fp16=False,
        no_pl_reg=False,
        cl_reg=False,
        fq_layers=[],
        fq_dict_size=256,
        attn_layers=[],
        no_const=False,
        aug_prob=0.,
        aug_types=['translation', 'cutout'],
        top_k_training=False,
        generator_top_k_gamma=0.99,
        generator_top_k_frac=0.5,
        dual_contrast_loss=False,
        dataset_aug_prob=0.,
        multi_gpus=False,
        calculate_fid_every=None,
        calculate_fid_num_images=12800,
        clear_fid_cache=False,
        seed=42,
        log=False,
<<<<<<< HEAD:stylex/cli.py

        # A global scale to the custom losses
        # TODO: Check if changing encoder learning rate is more appropriate
        #       than rescaling the reconstruction loss
        kl_scaling=1,
        rec_scaling=5,

        # Path to the classifier
        classifier_path="mnist.pth",

        # This shouldn't ever be changed since we're working with
        # binary classificiation.
        num_classes=2,  # TODO: Is 2 for faces gender.

        # If unspecified, use the Discriminator as an encoder.
        # This is the way to go if we want to be close to the original paper.
        # Check out debug_endocers.py for the names of classes if you still want
        # to use a different encoder.
        encoder_class=None,

        # This is for making the image results be results of the
        # image -> encoder -> generator pipeline
        # Set False if training a standard GAN or if you want to see
        # examples from a noise vector
        sample_from_encoder=True,  

        # Alternatively trains the model with the StylEx loss
        # and the regular StyleGAN loss. If False just trains
        # using the encoder
        alternating_training=True,

        # For now, I've made it so dataset_name='MNIST' automatically 
        # loads and rebalances a 1 vs all MNIST dataset.
        # TODO: Make custom dataloaders work in a distributed setting (low priority)
        dataset_name=None,


=======
        classifier_model_name="mnist.pth",  # TODO: Used to be FFHQ-Gender.pth
        classifier_classes=2,  # TODO: Is 2 for faces gender.
        sample_from_encoder=True,  # TODO: Default is False
>>>>>>> main:gan_training_og_repo/stylegan2_pytorch/cli.py
        tensorboard_dir="tb_logs_stylex",  # TODO: None for not logging
):

    model_args = dict(
        name=name,
        results_dir=results_dir,
        models_dir=models_dir,
        batch_size=batch_size,
        gradient_accumulate_every=gradient_accumulate_every,
        image_size=image_size,
        network_capacity=network_capacity,
        fmap_max=fmap_max,
        transparent=transparent,
        lr=learning_rate,
        lr_mlp=lr_mlp,
        ttur_mult=ttur_mult,
        rel_disc_loss=rel_disc_loss,
        num_workers=num_workers,
        save_every=save_every,
        evaluate_every=evaluate_every,
        num_image_tiles=num_image_tiles,
        trunc_psi=trunc_psi,
        fp16=fp16,
        no_pl_reg=no_pl_reg,
        cl_reg=cl_reg,
        fq_layers=fq_layers,
        fq_dict_size=fq_dict_size,
        attn_layers=attn_layers,
        no_const=no_const,
        aug_prob=aug_prob,
        aug_types=cast_list(aug_types),
        top_k_training=top_k_training,
        generator_top_k_gamma=generator_top_k_gamma,
        generator_top_k_frac=generator_top_k_frac,
        dual_contrast_loss=dual_contrast_loss,
        dataset_aug_prob=dataset_aug_prob,
        calculate_fid_every=calculate_fid_every,
        calculate_fid_num_images=calculate_fid_num_images,
        clear_fid_cache=clear_fid_cache,
        mixed_prob=mixed_prob,
        log=log,
<<<<<<< HEAD:stylex/cli.py
        kl_scaling=kl_scaling,
        rec_scaling=rec_scaling,
        classifier_path=classifier_path,
        num_classes=num_classes,
        encoder_class=encoder_class,
        dataset_name=dataset_name,
        sample_from_encoder=sample_from_encoder,
        alternating_training=alternating_training,
=======
        classifier_model_name=classifier_model_name,
        classifier_classes=classifier_classes,
        sample_from_encoder=sample_from_encoder,
>>>>>>> main:gan_training_og_repo/stylegan2_pytorch/cli.py
        tensorboard_dir=tensorboard_dir,
    )



    if generate:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        for num in tqdm(range(num_generate)):
            model.evaluate(f'{samples_name}-{num}', num_image_tiles)
        print(f'sample images generated at {results_dir}/{name}/{samples_name}')
        return

    if generate_interpolation:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        model.generate_interpolation(samples_name, num_image_tiles, num_steps=interpolation_num_steps,
                                     save_frames=save_frames)
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')
        return

    world_size = torch.cuda.device_count()

    if world_size == 1 or not multi_gpus:
        run_training(0, 1, model_args, data, load_from, new, num_train_steps, name, seed, dataset_name=dataset_name)
        return

    mp.spawn(run_training,
             args=(world_size, model_args, data, load_from, new, num_train_steps, name, seed),
             nprocs=world_size,
             join=True)


def main():
    fire.Fire(train_from_folder)


if __name__ == '__main__':
    main()
