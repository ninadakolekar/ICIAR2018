# -*- coding: utf-8 -*-
"""Defines network architectures for PW and IW Network
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BaseNetwork(nn.Module):
     '''
    Base Network class for initializing network weights and contain meta-information about the network
    Inherits from `torch.nn.Module` class

    Attributes:
        _name (str): Name of the network
        _channels (int): Number of channels in the output of the PW Network/input to the IW network
    '''

    def __init__(self, name, channels=1):
        ''' Initialises the class attributes 
        
        Args:
            name (str): Name of the network
            channels (int): Number of channels in the output of the PW Network/input to the IW network
        
        '''
        super(BaseNetwork, self).__init__()
        self._name = name
        self._channels = channels

    def name(self):
        ''' Return the name of the network
        
        Args:
            None
        
        Returns:
            Network name (str)
        
        '''
        return self._name

    def initialize_weights(self):
        ''' Initializes the network weights
        
        Args:
            None
        
        Returns:
            None
        
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class PatchWiseNetwork(BaseNetwork):
    '''
    Describes network architecture of the Patch-wise network.
    Inherits from the Base Network class.

    Attributes:
        features (torch.nn.Sequential): Sequential object encoding the architecture of PW Network to produce required feature-maps
        classifier (torch.nn.Sequential): Sequential object encoding the architecture of PW Network to compute prediction from feature-maps
    '''

    def __init__(self, channels=1,init=True):
        ''' Initialises the class attributes 
        
        Args:
            channels (int): Number of channels in the output of the PW Network
            init (bool): Boolean indicating whether to initialize the PatchWiseNetwork instance or not
        '''

        super(PatchWiseNetwork, self).__init__('pw' + str(channels), channels)

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=2,
                stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Block 2
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=2,
                stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 3
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=2,
                stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 4
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 5
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=256,
                out_channels=channels,
                kernel_size=1,
                stride=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(channels * 64 * 64, 3),
        )

        if init:
            self.initialize_weights()

    def forward(self, x):
        ''' Computes forward pass of the patch-wise network on given input  
        
        Args:
            x (`torch.Tensor`): Input Tesnor
        
        Returns:
            Softmax activations corresponding to each class (`torch.Tensor`)
        '''

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x


class ImageWiseNetwork(BaseNetwork):
    '''
    Describes network architecture of the Image-wise network.
    Inherits from the Base Network class.

    Attributes:
        features (torch.nn.Sequential): Sequential object encoding the architecture of PW Network to produce required feature-maps
        classifier (torch.nn.Sequential): Sequential object encoding the architecture of PW Network to compute prediction from feature-maps
    '''
    
    def __init__(self, channels=1,init=True):
        ''' Initialises the class attributes 
        
        Args:
            channels (int): Number of channels in the output of the PW Network
            init (bool): Boolean indicating whether to initialize the ImageWiseNetwork instance or not
        '''

        super(ImageWiseNetwork, self).__init__('iw' + str(channels), channels)

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=12 *
                channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=2,
                stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 2
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=2,
                stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=1,
                kernel_size=1,
                stride=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5, ),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5, ),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5, ),

            nn.Linear(64, 3),
        )

        if init:
            self.initialize_weights()

    def forward(self, x):
        ''' Computes forward pass of the image-wise network on given input  
        
        Args:
            x (`torch.Tensor`): Input Tesnor
        
        Returns:
            Softmax activations corresponding to each class (`torch.Tensor`)
        '''

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x
