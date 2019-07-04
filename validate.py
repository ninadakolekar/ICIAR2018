# -*- coding: utf-8 -*-
"""Defines validation procedure for Patch-wise and Image-wise network
"""

import matplotlib
matplotlib.use('Agg')

# PyTORCH PACKAGES

import torch

# SOURCE PACKAGES

from src.options import TrainingOptions
from src.networks import PatchWiseNetwork, ImageWiseNetwork
from src.models import PatchWiseModel, ImageWiseModel

if __name__ == "__main__":

    args = TrainingOptions().parse()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    pw_network = PatchWiseNetwork(args.channels,init=False)
    iw_network = ImageWiseNetwork(args.channels,init=False)

    if args.network == '0' or args.network == '1':
        pw_model = PatchWiseModel(args, pw_network)
        pw_model.validate()

    if args.network == '0' or args.network == '2':
        iw_model = ImageWiseModel(args, iw_network, pw_network)
        iw_model.validate(roc=True)
