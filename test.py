# -*- coding: utf-8 -*-
"""Defines test-time procedure for Patch-wise and Image-wise network
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

    if args.network == '1':
        pw_model = PatchWiseModel(args, pw_network)
        pw_model.test(args.testset_path, verbose=True)

    else:
        im_model = ImageWiseModel(args, iw_network, pw_network)
        im_model.test(args.testset_path, ensemble=args.ensemble == 1)
