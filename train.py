# -*- coding: utf-8 -*-
"""Defines training procedure for Patch-wise and Image-wise network
"""

# COMMAND
# python train.py --dataset-path /home/nitish/Desktop/ninad/data/
# --checkpoints-path ../pw1 --gpu-ids 0,1 --network 1 --batch-size 32
# --test-batch-size 32 --debug 1 --tid pw1

from src.models import PatchWiseModel, ImageWiseModel
from src.networks import PatchWiseNetwork, ImageWiseNetwork
from src.options import TrainingOptions
import torch
import matplotlib
matplotlib.use('Agg')

# PyTORCH PACKAGES


# SOURCE PACKAGES


if __name__ == "__main__":

    args = TrainingOptions().parse()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    pw_network = PatchWiseNetwork(args.channels)
    iw_network = ImageWiseNetwork(args.channels)

    if args.network == '0' or args.network == '1':
        pw_model = PatchWiseModel(args, pw_network)
        pw_model.train()

    if args.network == '0' or args.network == '2':
        iw_model = ImageWiseModel(args, iw_network, pw_network)
        iw_model.train()
