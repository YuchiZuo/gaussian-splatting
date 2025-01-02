#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, mask = None ):
    if mask is not None:
        mse = (((img1 - img2)) ** 2) * mask
        num_valid_elements = mask.view(mask.shape[0], -1).sum(1)
        num_valid_elements = torch.max(num_valid_elements, torch.tensor(1e-10).to(mask.device))  # 防止除以0
        mse = mse.view(mse.shape[0], -1).sum(1) / num_valid_elements
    else:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
