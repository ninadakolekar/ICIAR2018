# -*- coding: utf-8 -*-
"""Defines command-line arguments/options for training, test and validations
"""

from __future__ import print_function
import os
import argparse

import torch


class TrainingOptions(object):
    '''
    Defines command-line arguments/options for training and functions to parse them

    An object of this class constructs an argument parser with appropriate options for training of Patch-wise and Image-wise network

    Attributes:
        _parser: An object of the ArgumentParser class of the argparse module of the Python3 standard library
    '''

    def __init__(self):
        ''' Initialises the argument parser with appropriate options

        Args:
            None
        '''
        parser = argparse.ArgumentParser(
            description='Classification of morphology in cancer cell-lines')
        parser.add_argument(
            '--dataset-path',
            type=str,
            default='./dataset',
            help='dataset path (default: ./dataset)')
        parser.add_argument(
            '--testset-path',
            type=str,
            default='',
            help='file or directory address to the test set')
        parser.add_argument(
            '--checkpoints-path',
            type=str,
            default='./checkpoints',
            help='models are saved here')
        parser.add_argument(
            '--batch-size',
            type=int,
            default=64,
            metavar='N',
            help='input batch size for training (default: 64)')
        parser.add_argument(
            '--test-batch-size',
            type=int,
            default=64,
            metavar='N',
            help='input batch size for testing (default: 64)')
        parser.add_argument(
            '--patch-stride',
            type=int,
            default=256,
            metavar='N',
            help='How far the centers of two consecutive patches are in the image (default: 256)')
        parser.add_argument(
            '--epochs',
            type=int,
            default=30,
            metavar='N',
            help='number of epochs to train (default: 30)')
        parser.add_argument(
            '--lr',
            type=float,
            default=0.001,
            metavar='LR',
            help='learning rate (default: 0.01)')
        parser.add_argument(
            '--beta1',
            type=float,
            default=0.9,
            metavar='M',
            help='Adam beta1 (default: 0.9)')
        parser.add_argument(
            '--beta2',
            type=float,
            default=0.999,
            metavar='M',
            help='Adam beta2 (default: 0.999)')
        parser.add_argument(
            '--no-cuda',
            action='store_true',
            default=False,
            help='disables CUDA training')
        parser.add_argument(
            '--seed',
            type=int,
            default=1,
            metavar='S',
            help='random seed (default: 1)')
        parser.add_argument(
            '--log-interval',
            type=int,
            default=50,
            metavar='N',
            help='how many batches to wait before logging training status')
        parser.add_argument(
            '--gpu-ids',
            type=str,
            default='0',
            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument(
            '--ensemble',
            type=int,
            default='1',
            help='whether to use model ensemble on test-set prediction (default: 1)')
        parser.add_argument(
            '--network',
            type=str,
            default='0',
            help='train patch-wise network: 1, image-wise network: 2 or both: 0 (default: 0)')
        parser.add_argument(
            '--channels',
            type=int,
            default=1,
            help='number of channels created by the patch-wise network that feeds into the image-wise network (default: 1)')
        parser.add_argument('--debug', type=int, default=0,
                            help='debugging (default: 0)')
        parser.add_argument('--tid', type=str, required=False)
        self._parser = parser

    def parse(self):
        ''' Parses the arguments from the CLI

        Note:
            Also sets the GPU settings as mentioned in the CLI arguments

        Args:
            None

        Returns:
            dict: A dictionary containing the selected options for the training
        '''
        opt = self._parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        opt.cuda = not opt.no_cuda and torch.cuda.is_available()
        opt.debug = opt.debug != 0

        args = vars(opt)
        print('\n------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------\n')

        return opt


class ValidationOptions(object):
    '''
    Defines command-line arguments/options for validation and functions to parse them

    An object of this class constructs an argument parser with appropriate options for validation of Patch-wise and Image-wise network

    Attributes:
        _parser: An object of the ArgumentParser class of the argparse module of the Python3 standard library
    '''

    def __init__(self):
        ''' Initialises the argument parser with appropriate options

        Args:
            None
        '''

        parser = argparse.ArgumentParser(
            description='Classification of morphology in cancer cell-lines')

        parser.add_argument(
            '--testset-path',
            type=str,
            required='True',
            help='Path to test directory or file')
        parser.add_argument(
            '--val',
            action='store_true',
            default=False,
            help='Set mode to validation')
        parser.add_argument(
            '--checkpoints-path',
            type=str,
            required=True,
            help='Path to saved model checkpoints')
        parser.add_argument(
            '--no-cuda',
            action='store_true',
            default=False,
            help='Disables CUDA evaluation')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed (default: 42)')
        parser.add_argument(
            '--outdir',
            required=True,
            help='Directory to output CSV and ROC')
        parser.add_argument(
            '--verbose',
            action='store_true',
            default=False,
            help='Enables verbose evaluation')
        parser.add_argument(
            '--gpu-ids',
            type=str,
            default='0',
            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument(
            '--channels',
            type=int,
            default=1,
            help='number of channels created by the patch-wise network that feeds into the image-wise network (default: 1)')
        self._parser = parser

    def parse(self):
        ''' Parses the arguments from the CLI

        Note:
            Also sets the GPU settings as mentioned in the CLI arguments

        Args:
            None

        Returns:
            dict: A dictionary containing the selected options for the validation
        '''

        opt = self._parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        opt.cuda = not opt.no_cuda and torch.cuda.is_available()

        args = vars(opt)

        if opt.verbose:

            print('\n------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------\n')

        return opt


class TestOptions(object):
    '''
    Defines command-line arguments/options for test-time and functions to parse them

    An object of this class constructs an argument parser with appropriate options for test-time prediction of Patch-wise and Image-wise network

    Attributes:
        _parser: An object of the ArgumentParser class of the argparse module of the Python3 standard library
    '''

    def __init__(self):
        ''' Initialises the argument parser with appropriate options

        Args:
            None
        '''

        parser = argparse.ArgumentParser(
            description='Classification of morphology in cancer cell-lines')

        parser.add_argument(
            '--testset-path',
            type=str,
            required='True',
            help='Path to test directory or file')
        parser.add_argument(
            '--val',
            action='store_true',
            default=False,
            help='Set mode to validation')
        parser.add_argument(
            '--checkpoints-path',
            type=str,
            required=True,
            help='Path to saved model checkpoints')
        parser.add_argument(
            '--no-cuda',
            action='store_true',
            default=False,
            help='Disables CUDA evaluation')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed (default: 42)')
        parser.add_argument(
            '--out-csv',
            required=True,
            help='Path to output CSV')
        parser.add_argument(
            '--verbose',
            action='store_true',
            default=False,
            help='Enables verbose evaluation')
        parser.add_argument(
            '--gpu-ids',
            type=str,
            default='0',
            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument(
            '--channels',
            type=int,
            default=1,
            help='number of channels created by the patch-wise network that feeds into the image-wise network (default: 1)')
        self._parser = parser

    def parse(self):
        ''' Parses the arguments from the CLI

        Note:
            Also sets the GPU settings as mentioned in the CLI arguments

        Args:
            None

        Returns:
            dict: A dictionary containing the selected options for the validation
        '''

        opt = self._parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        opt.cuda = not opt.no_cuda and torch.cuda.is_available()

        args = vars(opt)

        if opt.verbose:

            print('\n------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------\n')

        return opt
