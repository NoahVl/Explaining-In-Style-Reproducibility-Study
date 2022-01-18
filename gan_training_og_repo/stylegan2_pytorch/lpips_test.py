"""
from time import sleep

import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
#loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

import torch
img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.zeros(1,3,64,64)
d = loss_fn_alex(img0, img1)
sleep(1000)
"""
import lpips
import torch
from torch import nn

#  We used a learning rate of 0.002 and reconstruction loss both
#  on the image (using LPIPS) and on the W-vector (using L1 loss),

lpips_loss = lpips.LPIPS(net="alex")
l1_loss = nn.L1Loss()

def lpips_normalize(images):
    return images / torch.max(images, dim=0) * 2 - 1

def reconstruction_loss(encoder_batch: torch.Tensor, generated_images: torch.Tensor, encoder: nn.Module, encoder_w: torch.Tensor):
    encoder_batch_norm = lpips_normalize(encoder_batch)
    generated_images_norm = lpips_normalize(generated_images)

    # LPIPS reconstruction loss
    loss = lpips_loss(encoder_batch_norm, generated_images_norm) + l1_loss(encoder_w, encoder(generated_images))

