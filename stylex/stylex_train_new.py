import os
import sys
import math
import fire
import json

import lpips
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
from contextlib import contextmanager, ExitStack

import numpy as np

import torch
from torch import nn, einsum
from torch.utils import data
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange, repeat
from kornia.filters import filter2d

import torchvision
from torchvision import transforms
from version import __version__
from diff_augment import DiffAugment

from vector_quantize_pytorch import VectorQuantize

from PIL import Image
from pathlib import Path

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

import aim

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# Classifier
from mobilenet_classifier import MobileNet
from resnet_classifier import ResNet

# Encoders for debugging or additional testing
import debug_encoders

# constants

NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']


# helper classes

class NanException(Exception):
    pass


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ChanNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, *_, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


# attention

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding=0, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.nonlin = nn.GELU()
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding=1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=h), (q, k, v))

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, x=x, y=y)

        out = self.nonlin(out)
        return self.to_out(out)


# one layer of self-attention and feedforward, for images

attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(PreNorm(chan, LinearAttention(chan))),
    Residual(PreNorm(chan, nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])


# helpers


def make_weights_for_balanced_classes(dataset, nclasses):
    count = [0] * nclasses

    print(count)
    for item in dataset:

        if len(item) == 1:
            print(item)
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(dataset)
    for idx, val in enumerate(dataset):
        weight[idx] = weight_per_class[val[1]]
    return weight


def exists(val):
    return val is not None


@contextmanager
def null_context():
    yield


def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]

    return multi_contexts


def default(value, d):
    return value if exists(value) else d


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def cast_list(el):
    return el if isinstance(el, list) else [el]


def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return not exists(t)


def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException


def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts = head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield


def loss_backwards(fp16, loss, optimizer, loss_id, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


def gradient_penalty(images, output, weight=10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def calc_pl_lengths(styles, images):
    device = images.device
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape, device=device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()


def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)


def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]


def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)


def latent_to_w(style_vectorizer, latent_descr, probabilities):
    return [(torch.cat((style_vectorizer(z), probabilities), dim=1), num_layers) for z, num_layers in latent_descr]


def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda(device)


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


def lpips_normalize(images):
    """
    Function that scales images to be in range [-1, 1] (needed for LPIPS alex model)
    """
    images_flattened = images.view(images.shape[0], -1)
    _max = images_flattened.max(dim=1)[0].view(-1, 1, 1, 1)
    _min = images_flattened.min(dim=1)[0].view(-1, 1, 1, 1)
    return (images - _min) / (_max - _min) * 2 - 1


# losses

def gen_hinge_loss(fake, real):
    return fake.mean()


def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()


def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i=t1.shape[0])
        t = torch.cat((t1, t2), dim=-1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device=device, dtype=torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)


# Our losses
lpips_loss = lpips.LPIPS(net="alex").cuda(0)  # image should be RGB, IMPORTANT: normalized to [-1,1]
l1_loss = nn.L1Loss()
kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)


def reconstruction_loss(encoder_batch: torch.Tensor, generated_images: torch.Tensor, generated_images_w: torch.Tensor,
                        encoder_w: torch.Tensor):
    encoder_batch_norm = lpips_normalize(encoder_batch)
    generated_images_norm = lpips_normalize(generated_images)

    # LPIPS reconstruction loss
    loss = 0.1 * lpips_loss(encoder_batch_norm, generated_images_norm).mean() +\
           0.1 * l1_loss(encoder_w, generated_images_w) +\
           1 * l1_loss(encoder_batch, generated_images)
    return loss


def classifier_kl_loss(real_classifier_logits, fake_classifier_logits):
    # Convert logits to log_softmax and then KL loss

    # Get probabilities through softmax

    # real_probabilities = torch.softmax(real_classifier_logits, dim=1)
    # fake_probabilities = torch.softmax(fake_classifier_logits, dim=1)

    # if real_probabilities[0, 0] > real_probabilities[0, 1]:
    #     print('!!!!!!!!!')
    # else:
    #     print('?????????')

    real_classifier_probabilities = F.log_softmax(real_classifier_logits, dim=1)
    fake_classifier_probabilities = F.log_softmax(fake_classifier_logits, dim=1)

    loss = kl_loss(fake_classifier_probabilities, real_classifier_probabilities)

    return loss


# dataset

def convert_rgb_to_transparent(image):
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image


def convert_transparent_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))


def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image


# Right now, opting for a 'named' dataset will replace the usual dataloader
# used in training StylEx. It doesn't inherently support distributed
# training, but we aren't doing that yet and MNIST can be trained fairly
# easily on one GPU so it's not too much of a problem

# pass --dataset_name MNIST to cli.py to automatically sample from MNIST
# It will oversample the 'true' digit correctly

class MNIST_1vA(torch.utils.data.Dataset):
    def __init__(self, folder='./', digit=8):
        self.image_size = 32

        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

        self.dataset = torchvision.datasets.MNIST(folder, train=True, download=True, transform=self.transform)
        self.dataset.targets = self.dataset.targets == digit

    def __getlen__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]

        return image

    def __len__(self):
        return len(self.dataset)


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, transparent=False, aug_prob=0.):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)),
                        transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(transparent))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


# augmentations

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))


class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, probabilities=torch.Tensor([0.0, 0.0]), prob=0., types=[], detach=False):
        if random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=types)

        if detach:
            images = images.detach()

        return self.D(images, probabilities=probabilities)


# stylegan2 classes

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul=0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba=False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps=1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x


class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample=True, upsample_rgb=True, rgba=False):
        super().__init__()

        self.input_channels = input_channels
        self.filters = filters

        self.num_style_coords = self.input_channels + self.filters

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if exists(self.upsample):
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)

        # Perturb here

        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)

        style_coords = torch.cat([style1, style2], dim=-1)

        # Perturb here

        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)

        return x, rgb, style_coords


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2d(filters, filters, 3, padding=1, stride=2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity=16, transparent=False, attn_layers=[], no_const=False,
                 fmap_max=512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])

        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
                rgba=transparent
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise, get_style_coords=False):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        if get_style_coords:
            style_coords_list = []

        for style, block, attn in zip(styles, self.blocks, self.attns):
            if exists(attn):
                x = attn(x)

            x, rgb, style_coords = block(x, rgb, style, input_noise)

            if get_style_coords:
                style_coords_list.append(style_coords)

        if get_style_coords:
            style_coords = torch.cat(style_coords_list, dim=1)
            return rgb, style_coords

        else:
            return rgb

    def get_style_coords(self, styles):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)


class DiscriminatorE(nn.Module):
    def __init__(self, image_size, network_capacity=16, fq_layers=[], fq_dict_size=256, attn_layers=[],
                 transparent=False, encoder=False, encoder_dim=512, fmap_max=512):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.encoder = encoder
        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.encoder_dim = encoder_dim

        if not self.encoder:
            self.fc = nn.Linear(latent_dim, 2)
        else:
            self.fc = nn.Linear(latent_dim, self.encoder_dim)

    def forward(self, x, probabilities=torch.Tensor([0.0, 0.0])):
        b, *_ = x.shape

        # quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            x = block(x)

            if exists(attn_block):
                x = attn_block(x)

            # if exists(q_block):
            #     x, loss = q_block(x)
            #     quantize_loss += loss

        x = self.final_conv(x)
        x = self.flatten(x)

        x = self.fc(x)

        if not self.encoder:
            # Do a weighted sum of x[:, 0] and x[:, 1] with probabilities
            x = x[:, 0] * probabilities[:, 0] + x[:, 1] * probabilities[:, 1]

        return x.squeeze()  # , quantize_loss


class StylEx(nn.Module):
    def __init__(self, image_size, latent_dim=514, fmap_max=512, style_depth=8, network_capacity=16, transparent=False,
                 fp16=False, cl_reg=False, steps=1, lr=1e-4, ttur_mult=2, fq_layers=[], fq_dict_size=256,
                 attn_layers=[], no_const=False, lr_mlp=0.1, rank=0, classifier_labels=2, encoder_class=None):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        # DiscriminatorEncoder is default unless specified by encoder class name

        if encoder_class is None:
            self.encoder = DiscriminatorE(image_size, network_capacity, encoder=True, fq_layers=fq_layers,
                                          fq_dict_size=fq_dict_size,
                                          attn_layers=attn_layers, transparent=transparent, fmap_max=fmap_max)
        else:
            self.encoder = debug_encoders.encoder_dict[encoder_class]

        # Fixed
        self.num_classes = 2

        self.S = StyleVectorizer(latent_dim - self.num_classes, style_depth, lr_mul=lr_mlp)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent=transparent, attn_layers=attn_layers,
                           no_const=no_const, fmap_max=fmap_max)
        self.D = DiscriminatorE(image_size, network_capacity, fq_layers=fq_layers, fq_dict_size=fq_dict_size,
                                attn_layers=attn_layers, transparent=transparent, fmap_max=fmap_max)

        self.SE = StyleVectorizer(latent_dim - self.num_classes, style_depth, lr_mul=lr_mlp)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent=transparent, attn_layers=attn_layers,
                            no_const=no_const)

        self.D_cl = None

        # Is turned off by default
        if cl_reg:
            from contrastive_learner import ContrastiveLearner
            # experimental contrastive loss discriminator regularization
            assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        # turn off grad for exponential moving averages
        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        # init optimizers
        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = Adam([{'params': generator_params}, {'params': list(self.encoder.parameters()), 'lr': 1e-5}],
                          lr=self.lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr=self.lr * ttur_mult, betas=(0.5, 0.9))

        # init weights
        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda(rank)

        # startup apex mixed precision
        self.fp16 = fp16
        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE, self.encoder), (self.G_opt, self.D_opt) = amp.initialize(
                [self.S, self.G, self.D, self.SE, self.GE, self.encoder], [self.G_opt, self.D_opt], opt_level='O1',
                num_losses=3)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x


class Trainer():
    def __init__(
            self,
            name='default',
            results_dir='results',
            models_dir='models',
            base_dir='./',
            image_size=128,
            network_capacity=16,
            fmap_max=512,
            transparent=False,
            batch_size=4,
            mixed_prob=0.9,
            gradient_accumulate_every=1,
            lr=2e-4,
            lr_mlp=0.1,
            ttur_mult=2,
            rel_disc_loss=False,
            num_workers=None,
            save_every=1000,
            evaluate_every=1000,
            num_image_tiles=8,
            trunc_psi=0.6,
            fp16=False,
            cl_reg=False,
            no_pl_reg=False,
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
            calculate_fid_every=None,
            calculate_fid_num_images=12800,
            clear_fid_cache=False,
            is_ddp=False,
            rank=0,
            world_size=1,
            log=False,
            kl_scaling=1,
            rec_scaling=10,
            classifier_path="mnist.pth",  # TODO: Used to be FFHQ-Gender.pth,
            num_classes=2,  # TODO: Used to be 2 for faces gender.
            encoder_class=None,
            kl_rec_during_disc=False,
            alternating_training=True,
            sample_from_encoder=False,
            dataset_name=None,
            tensorboard_dir=None,
            classifier_name=None,
            *args,
            **kwargs
    ):
        self.model_params = [args, kwargs]
        self.StylEx = None

        self.kl_scaling = kl_scaling
        self.rec_scaling = rec_scaling

        self.kl_rec_during_disc = kl_rec_during_disc

        self.alternating_training = alternating_training

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.fmap_max = fmap_max
        self.transparent = transparent

        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size
        self.has_fq = len(self.fq_layers) > 0

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.rel_disc_loss = rel_disc_loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.num_image_tiles = num_image_tiles
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.no_pl_reg = no_pl_reg
        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.total_rec_loss = 0
        self.total_kl_loss = 0
        self.q_loss = None
        self.last_gp_loss = None
        self.last_cr_loss = None
        self.last_fid = None

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        self.top_k_training = top_k_training
        self.generator_top_k_gamma = generator_top_k_gamma
        self.generator_top_k_frac = generator_top_k_frac

        self.dual_contrast_loss = dual_contrast_loss

        assert not (is_ddp and cl_reg), 'Contrastive loss regularization does not work well with multi GPUs yet'
        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size
        self.sample_from_encoder = sample_from_encoder

        self.logger = aim.Session(experiment=name) if log else None

        if self.alternating_training:

            # multiply losses by 2 since they are only calculated every other iteration if using alternating training
            self.rec_scaling *= 2
            self.kl_scaling *= 2
        else:
            encoder_input = True

        # Load classifier
        self.num_classes = num_classes
        self.classifier = None
        if classifier_name.lower() == "resnet":
            self.classifier = ResNet(classifier_path, cuda_rank=rank, output_size=self.num_classes,
                                     image_size=image_size)
        else:
            self.classifier = MobileNet(classifier_path, cuda_rank=rank, output_size=self.num_classes,
                                        image_size=image_size)  # Automatically put into eval mode

        # Load tensorboard, create writer
        self.tb_writer = None
        if exists(tensorboard_dir):
            self.tb_writer = SummaryWriter(os.path.join(tensorboard_dir, name))

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}

    def init_StylEx(self):
        args, kwargs = self.model_params
        self.StylEx = StylEx(lr=self.lr, lr_mlp=self.lr_mlp, ttur_mult=self.ttur_mult, image_size=self.image_size,
                             network_capacity=self.network_capacity, fmap_max=self.fmap_max,
                             transparent=self.transparent, fq_layers=self.fq_layers, fq_dict_size=self.fq_dict_size,
                             attn_layers=self.attn_layers, fp16=self.fp16, cl_reg=self.cl_reg, no_const=self.no_const,
                             rank=self.rank, classifier_labels=self.num_classes, *args, **kwargs)

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank]}
            self.S_ddp = DDP(self.StylEx.S, **ddp_kwargs)
            self.G_ddp = DDP(self.StylEx.G, **ddp_kwargs)
            self.D_ddp = DDP(self.StylEx.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.StylEx.D_aug, **ddp_kwargs)

        if exists(self.logger):
            self.logger.set_params(self.hparams)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.fmap_max = config.pop('fmap_max', 512)
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        self.lr_mlp = config.pop('lr_mlp', 0.1)
        del self.StylEx
        self.init_StylEx()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'lr_mlp': self.lr_mlp,
                'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size,
                'attn_layers': self.attn_layers, 'no_const': self.no_const}

    def set_data_src(self, folder='./', dataset_name=None):
        if dataset_name is None:
            self.dataset = Dataset(folder, self.image_size, transparent=self.transparent,
                                   aug_prob=self.dataset_aug_prob)
            num_workers = num_workers = default(self.num_workers, NUM_CORES if not self.is_ddp else 0)

            sampler = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size,
                                         shuffle=True) if self.is_ddp else None

            dataloader = data.DataLoader(self.dataset, num_workers=num_workers,
                                         batch_size=math.ceil(self.batch_size / self.world_size), sampler=sampler,
                                         shuffle=not self.is_ddp, drop_last=True, pin_memory=True)

        if dataset_name == 'MNIST':
            self.dataset = MNIST_1vA(digit=8)

            weights = make_weights_for_balanced_classes(self.dataset.dataset, self.num_classes)
            sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

            dataloader = data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler)

        self.loader = cycle(dataloader)

        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if not exists(self.StylEx):
            self.init_StylEx()

        self.StylEx.encoder.train()
        self.StylEx.train()
        total_disc_loss = torch.tensor(0.).cuda(self.rank)
        total_gen_loss = torch.tensor(0.).cuda(self.rank)
        total_rec_loss = torch.tensor(0.).cuda(self.rank)
        total_kl_loss = torch.tensor(0.).cuda(self.rank)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.StylEx.G.image_size
        latent_dim = self.StylEx.G.latent_dim
        num_layers = self.StylEx.G.num_layers

        aug_prob = self.aug_prob
        aug_types = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = not self.no_pl_reg and self.steps > 5000 and self.steps % 32 == 0
        apply_cl_reg_to_generated = self.steps > 20000

        S = self.StylEx.S if not self.is_ddp else self.S_ddp
        G = self.StylEx.G if not self.is_ddp else self.G_ddp
        D = self.StylEx.D if not self.is_ddp else self.D_ddp
        D_aug = self.StylEx.D_aug if not self.is_ddp else self.D_aug_ddp

        backwards = partial(loss_backwards, self.fp16)

        # setup losses

        if not self.dual_contrast_loss:
            D_loss_fn = hinge_loss
            G_loss_fn = gen_hinge_loss
            G_requires_reals = False
        else:
            D_loss_fn = dual_contrastive_loss
            G_loss_fn = dual_contrastive_loss
            G_requires_reals = True

        # train discriminator

        avg_pl_length = self.pl_mean
        self.StylEx.D_opt.zero_grad()

        if self.alternating_training:
            encoder_input = False
        else:
            encoder_input = True

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, S, G]):
            discriminator_batch = next(self.loader).cuda(self.rank)
            discriminator_batch.requires_grad_()

            encoder_batch = next(self.loader).cuda(self.rank)
            encoder_batch.requires_grad_()

            encoder_batch_logits = self.classifier.classify_images(encoder_batch)
            encoder_batch_probabilities = F.softmax(encoder_batch_logits, dim=1)

            if not self.alternating_training or encoder_input:
                self.StylEx.encoder.zero_grad()

                encoder_output = self.StylEx.encoder(encoder_batch)

                # print(real_classified_logits[:,0] > real_classified_logits[:,1])

                style = [(torch.cat((encoder_output, encoder_batch_probabilities), dim=1),
                          self.StylEx.G.num_layers)]  # Has to be bracketed because expects a noise mix
                noise = image_noise(batch_size, image_size, device=self.rank)

                w_styles = styles_def_to_tensor(style)
            else:
                get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
                style = get_latents_fn(batch_size, num_layers, latent_dim - self.num_classes, device=self.rank)
                noise = image_noise(batch_size, image_size, device=self.rank)

                w_space = latent_to_w(S, style, encoder_batch_probabilities)
                w_styles = styles_def_to_tensor(w_space)

            #  Ah, that's a good point, we forgot to mention that fact in the paper. We folow standard procedure of conditional GAN (see for example in the BigGAN paper)
            #  and multiply the output of the discriminator with the conditional input.
            #  That is used to give the discriminator information of the input class which it can use to predict whether it's a image belonging to this class or not.

            # We experimented with both methods, and found that concatenating the labels to W works better.
            # These logits are sampled from the dataset - in fact we take the same logits which are used by the encoder in the same step.

            generated_images = G(w_styles, noise)

            fake_output = D_aug(generated_images.clone().detach(), probabilities=encoder_batch_probabilities,
                                detach=True, **aug_kwargs)
            real_output = D_aug(discriminator_batch, probabilities=encoder_batch_probabilities, **aug_kwargs)

            real_output_loss = real_output
            fake_output_loss = fake_output

            if self.rel_disc_loss:
                real_output_loss = real_output_loss - fake_output.mean()
                fake_output_loss = fake_output_loss - real_output.mean()

            divergence = D_loss_fn(real_output_loss, fake_output_loss)
            disc_loss = divergence

            if self.has_fq:
                quantize_loss = (fake_q_loss + real_q_loss).mean()
                self.q_loss = float(quantize_loss.detach().item())

                disc_loss = disc_loss + quantize_loss

            if apply_gradient_penalty:
                gp = gradient_penalty(discriminator_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                self.track(self.last_gp_loss, 'GP')
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every

            if self.alternating_training and encoder_input:
                if self.kl_rec_during_disc:
                    generated_images_w = self.StylEx.encoder(generated_images)
                    rec_loss = self.rec_scaling * reconstruction_loss(encoder_batch, generated_images,
                                                                      generated_images_w,
                                                                      encoder_output) / self.gradient_accumulate_every

                    gen_image_classified_logits = self.classifier.classify_images(generated_images)

                    kl_loss = self.kl_scaling * classifier_kl_loss(encoder_batch_logits,
                                                                   gen_image_classified_logits) / self.gradient_accumulate_every
                    # rec_loss = rec_loss / self.ttur_scaling

                    backwards(rec_loss, self.StylEx.G_opt, loss_id=3)
                    total_rec_loss += rec_loss.detach().item()  # / self.gradient_accumulate_every

                    backwards(kl_loss, self.StylEx.G_opt, loss_id=4)

                    total_kl_loss += kl_loss.detach().item()  # / self.gradient_accumulate_every

                encoder_input = False
            elif self.alternating_training:
                encoder_input = True

            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.StylEx.D_opt, loss_id=1)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.track(self.d_loss, 'D')

        self.StylEx.D_opt.step()

        # train generator

        if self.alternating_training:
            encoder_input = False

        self.StylEx.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[S, G, D_aug]):
            image_batch = next(self.loader).cuda(self.rank)
            image_batch.requires_grad_()

            encoder_batch_logits = self.classifier.classify_images(image_batch)
            encoder_batch_probabilities = F.softmax(encoder_batch_logits, dim=1)

            if not self.alternating_training or encoder_input:
                encoder_output = self.StylEx.encoder(image_batch)

                style = [(torch.cat((encoder_output, encoder_batch_probabilities), dim=1), self.StylEx.G.num_layers)]
                noise = image_noise(batch_size, image_size, device=self.rank)

                w_styles = styles_def_to_tensor(style)
            else:
                style = get_latents_fn(batch_size, num_layers, latent_dim - self.num_classes, device=self.rank)
                noise = image_noise(batch_size, image_size, device=self.rank)

                w_space = latent_to_w(S, style, encoder_batch_probabilities)
                w_styles = styles_def_to_tensor(w_space)

            generated_images = G(w_styles, noise)
            gen_image_classified_logits = self.classifier.classify_images(generated_images)

            fake_output = D_aug(generated_images, probabilities=encoder_batch_probabilities, **aug_kwargs)
            fake_output_loss = fake_output

            real_output = None
            if G_requires_reals:
                image_batch = next(self.loader).cuda(self.rank)
                real_output = D_aug(image_batch, detach=True, **aug_kwargs)
                real_output = real_output.detach()

            if self.top_k_training:
                epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
                k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
                k = math.ceil(batch_size * k_frac)

                if k != batch_size:
                    fake_output_loss = fake_output_loss.topk(k=k, largest=False)

            # Our losses
            if not self.alternating_training or encoder_input:
                rec_loss = self.rec_scaling * reconstruction_loss(image_batch, generated_images,
                                                                  self.StylEx.encoder(generated_images),
                                                                  encoder_output) / self.gradient_accumulate_every
                kl_loss = self.kl_scaling * classifier_kl_loss(encoder_batch_logits,
                                                               gen_image_classified_logits) / self.gradient_accumulate_every

                # < > *  2 ** (num of iters)

            # Original loss
            loss = G_loss_fn(fake_output_loss, real_output)
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)

            if not self.alternating_training or encoder_input:
                added_losses = gen_loss + rec_loss + kl_loss

                backwards(added_losses, self.StylEx.G_opt, loss_id=2)
                # backwards(gen_loss, self.StylEx.G_opt, loss_id=2, retain_graph=True)
                # backwards(rec_loss, self.StylEx.G_opt, loss_id=3, retain_graph=True)
                # backwards(kl_loss, self.StylEx.G_opt, loss_id=4)

                total_gen_loss += loss.detach().item() / self.gradient_accumulate_every
                total_rec_loss += rec_loss.detach().item()

                total_kl_loss += kl_loss.detach().item()

                self.g_loss = float(total_gen_loss)
                self.total_rec_loss = float(total_rec_loss)
                self.total_kl_loss = float(total_kl_loss)
            else:
                backwards(gen_loss, self.StylEx.G_opt, loss_id=2)

                total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

                self.g_loss = float(total_gen_loss)

            encoder_input = not encoder_input

        # If writer exists, write losses
        if exists(self.tb_writer):
            self.tb_writer.add_scalar('loss/G', self.g_loss, self.steps)
            self.tb_writer.add_scalar('loss/D', self.d_loss, self.steps)
            self.tb_writer.add_scalar('loss/rec', self.total_rec_loss, self.steps)
            self.tb_writer.add_scalar('loss/kl', self.total_kl_loss, self.steps)

        # If writer exists, write losses
        if exists(self.tb_writer):
            self.tb_writer.add_scalar('loss/G', self.g_loss, self.steps)
            self.tb_writer.add_scalar('loss/D', self.d_loss, self.steps)
            self.tb_writer.add_scalar('loss/rec', self.total_rec_loss, self.steps)
            self.tb_writer.add_scalar('loss/kl', self.total_kl_loss, self.steps)

        self.track(self.g_loss, 'G')
        self.track(self.total_rec_loss, 'Rec')
        self.track(self.total_kl_loss, 'KL')

        self.StylEx.G_opt.step()

        # calculate moving averages

        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
            self.track(self.pl_mean, 'PL')

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.StylEx.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.StylEx.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                self.evaluate(encoder_input=self.sample_from_encoder, num=floor(self.steps / self.evaluate_every))

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(self.calculate_fid_num_images / self.batch_size)
                fid = self.calculate_fid(num_batches)
                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid}\n')

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, encoder_input=False, num=0, trunc=1.0):
        self.StylEx.eval()
        # ext = self.image_extension  TODO: originally only png if self.transparency was enabled
        ext = "png"
        num_rows = self.num_image_tiles

        latent_dim = self.StylEx.G.latent_dim
        image_size = self.StylEx.G.image_size
        num_layers = self.StylEx.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim - self.num_classes, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        # regular
        from_encoder_string = ""
        image_batch = None

        if encoder_input:
            from_encoder_string = "from_encoder"
            image_batch = next(self.loader).cuda(self.rank)

            with torch.no_grad():
                probabilities = F.softmax(self.classifier.classify_images(image_batch), dim=1)
                # w = [(torch.cat((self.StylEx.encoder(image_batch), real_classified_probabilities), dim=1), num_layers)]

                w_without_probabilities = [(self.StylEx.encoder(image_batch), num_layers)]

            num_rows = len(image_batch)
        else:
            w_without_probabilities = None

            # We want a [num_rows ** 2, 2] tensor of random probabilities
            probabilities = torch.rand(num_rows ** 2, 2, device=self.rank)
            probabilities = probabilities / torch.sum(probabilities, dim=1, keepdim=True)

        # pass images here

        generated_images = self.generate_truncated(self.StylEx.S, self.StylEx.G, latents, n, w=w_without_probabilities,
                                                   probabilities=probabilities, trunc_psi=self.trunc_psi)
        torchvision.utils.save_image(torch.cat((image_batch, generated_images)),
                                     str(self.results_dir / self.name / f'{str(num)}-{from_encoder_string}.{ext}'),
                                     nrow=num_rows)

        # moving averages

        generated_images = self.generate_truncated(self.StylEx.SE, self.StylEx.GE, latents, n,
                                                   w=w_without_probabilities, probabilities=probabilities,
                                                   trunc_psi=self.trunc_psi)
        torchvision.utils.save_image(torch.cat((image_batch, generated_images)),
                                     str(self.results_dir / self.name / f'{str(num)}-{from_encoder_string}-ema.{ext}'),
                                     nrow=num_rows)

        # mixing regularities

        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda(self.rank)
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim - self.num_classes, device=self.rank)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        probabilities = torch.rand(num_rows ** 2, 2, device=self.rank)
        probabilities = probabilities / torch.sum(probabilities, dim=1, keepdim=True)

        generated_images = self.generate_truncated(self.StylEx.SE, self.StylEx.GE, mixed_latents, n,
                                                   probabilities=probabilities, trunc_psi=self.trunc_psi)
        torchvision.utils.save_image(torch.cat((image_batch, generated_images)),
                                     str(self.results_dir / self.name / f'{str(num)}-{from_encoder_string}-mr.{ext}'),
                                     nrow=num_rows)

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        from pytorch_fid import fid_score
        torch.cuda.empty_cache()

        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        # remove any existing files used for fid calculation and recreate directories

        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader)
                for k, image in enumerate(real_batch.unbind(0)):
                    filename = str(k + batch_num * self.batch_size)
                    torchvision.utils.save_image(image, str(real_path / f'{filename}.png'))

        # generate a bunch of fake images in results / name / fid_fake

        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.StylEx.eval()
        ext = self.image_extension

        latent_dim = self.StylEx.G.latent_dim
        image_size = self.StylEx.G.image_size
        num_layers = self.StylEx.G.num_layers

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(self.batch_size, image_size, device=self.rank)

            # moving averages
            generated_images = self.generate_truncated(self.StylEx.SE, self.StylEx.GE, latents, noise,
                                                       trunc_psi=self.trunc_psi)

            for j, image in enumerate(generated_images.unbind(0)):
                torchvision.utils.save_image(image,
                                             str(fake_path / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))

        return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, noise.device, 2048)

    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi=0.75):
        S = self.StylEx.S
        batch_size = self.batch_size
        latent_dim = self.StylEx.G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim - self.num_classes, device=self.rank)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis=0)
            self.av = np.expand_dims(self.av, axis=0)

        av_torch = torch.from_numpy(self.av).cuda(self.rank)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi=0.75):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, trunc_psi=trunc_psi)
            w_space.append(tensor)
        return w_space

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, w=None, trunc_psi=0.75, probabilities=None, num_image_tiles=8):
        if w is None:
            w = map(lambda t: (S(t[0]), t[1]), style)

        w_truncated = self.truncate_style_defs(w, trunc_psi=trunc_psi)

        w_truncated = [(torch.cat((w_truncated[0], probabilities), dim=1), self.StylEx.G.num_layers)]

        w_styles = styles_def_to_tensor(w_truncated)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num=0, num_image_tiles=8, trunc=1.0, num_steps=100, save_frames=False):
        self.StylEx.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.StylEx.G.latent_dim
        image_size = self.StylEx.G.image_size
        num_layers = self.StylEx.G.num_layers

        # latents and noise

        image_batch = next(self.loader).cuda(self.rank)
        image_batch.requires_grad_()

        real_classified_logits = self.classifier.classify_images(image_batch[0])

        latents_low = noise(num_rows ** 2, latent_dim, device=self.rank)
        latents_high = noise(num_rows ** 2, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(torch.cat((interp_latents, real_classified_logits), dim=1), num_layers)]
            generated_images = self.generate_truncated(self.StylEx.SE, self.StylEx.GE, latents, n,
                                                       trunc_psi=self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow=num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())

            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)

            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:],
                       duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('PL', self.pl_mean),
            ('CR', self.last_cr_loss),
            ('Q', self.q_loss),
            ('FID', self.last_fid),
            ('Rec', self.total_rec_loss),
            ('KL', self.total_kl_loss)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def track(self, value, name):
        if not exists(self.logger):
            return
        self.logger.track(value, name=name)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'StylEx': self.StylEx.state_dict(),
            'version': __version__
        }

        if self.StylEx.fp16:
            save_data['amp'] = amp.state_dict()

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num=-1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        try:
            self.StylEx.load_state_dict(load_data['StylEx'])
        except Exception as e:
            print(
                'unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e
        if self.StylEx.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'])


class ModelLoader:
    def __init__(self, *, base_dir, name='default', load_from=-1):
        self.model = Trainer(name=name, base_dir=base_dir)
        self.model.load(load_from)

    def noise_to_styles(self, noise, trunc_psi=None):
        noise = noise.cuda()
        w = self.model.StylEx.SE(noise)
        if exists(trunc_psi):
            w = self.model.truncate_style(w)
        return w

    def styles_to_images(self, w):
        batch_size, *_ = w.shape
        num_layers = self.model.StylEx.GE.num_layers
        image_size = self.model.image_size
        w_def = [(w, num_layers)]

        w_tensors = styles_def_to_tensor(w_def)
        noise = image_noise(batch_size, image_size, device=0)

        images = self.model.StylEx.GE(w_tensors, noise)
        images.clamp_(0., 1.)
        return images
