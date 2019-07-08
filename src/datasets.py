# -*- coding: utf-8 -*-
"""Defines training, test and validation Dataset classes for Patch-wise and Image-wise networks
"""

import os
import glob
from pathlib import Path

# PyTORCH PACKAGES

import torch
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# SOURCE PACKAGES

from .patch_extractor import PatchExtractor
from src.config import LABELS, IMAGE_SIZE, PATCH_SIZE

class PatchWiseDataset(Dataset):
    '''
    Dataset class for input to the patch-wise network

    An object of this class loads the extracted patches and the label for each image from the given directory.
    Used for patch-wise network training.

    Attributes:
        path (str): Path of the train/test/val directory
        stride (int): Stride used to extract patches
        labels (dict): A dictionary with keys as filepaths to images and values as associated labels
        names (list): A list containing filepath of all images in an alphabetical order
        shape (tuple): A tuple describing shape of the dataset after all augmentations
        augment_size (int): Denotes the number of augmented images added to the dataset correponding to a single example
    '''
    def __init__(
            self,
            path,
            stride=PATCH_SIZE,
            rotate=False,
            flip=False,
            enhance=False):
        ''' Initialises the class attributes 
        
        Args:
            path (str): Path of the train/test/val directory
            stride (int): Stride used to extract patches (defaults to `ICIAR2018.src.config.PATCH_SIZE` in `ICIAR2018.src.config` Module)
            rotate (bool): A boolean indicating whether to use rotation for augmentation or not
            flip (bool): A boolean indicating whether to use flipping for augmentation or not
            enhance (bool): A boolean indicating whether to use color enhancement for augmentation or not
        
        '''
        super().__init__()

        wp = int((IMAGE_SIZE[0] - PATCH_SIZE) / stride + 1)
        hp = int((IMAGE_SIZE[1] - PATCH_SIZE) / stride + 1)
        labels = {
            name: index for index in range(
                len(LABELS)) for name in glob.glob(
                path +
                '/' +
                LABELS[index] +
                '/*.JPG')}

        self.path = path
        self.stride = stride
        self.labels = labels
        self.names = list(sorted(labels.keys()))
        # (files, x_patches, y_patches, rotations, flip, enhance)
        self.shape = (
            len(labels),
            wp,
            hp,
            (4 if rotate else 1),
            (2 if flip else 1),
            (2 if enhance else 1))
        self.augment_size = np.prod(self.shape) / len(labels)

    def __getitem__(self, index):
        ''' Fetches an example of given index from the dataset 
        
        Args:
            index (int): Index of the image in the dataset

        Returns:
            tensor: A PyTorch Tensor containing extracted patches from the image
            int: An integer denoting the associated class/label
        '''
        im, xpatch, ypatch, rotation, flip, enhance = np.unravel_index(
            index, self.shape)

        with Image.open(self.names[im]) as img:
            extractor = PatchExtractor(
                img=img, patch_size=PATCH_SIZE, stride=self.stride)
            patch = extractor.extract_patch((xpatch, ypatch))

            if rotation != 0:
                patch = patch.rotate(rotation * 90)

            if flip != 0:
                patch = patch.transpose(Image.FLIP_LEFT_RIGHT)

            if enhance != 0:
                factors = np.random.uniform(.5, 1.5, 3)
                patch = ImageEnhance.Color(patch).enhance(factors[0])
                patch = ImageEnhance.Contrast(patch).enhance(factors[1])
                patch = ImageEnhance.Brightness(patch).enhance(factors[2])

            label = self.labels[self.names[im]]
            return transforms.ToTensor()(patch), label

    def __len__(self):
        ''' Fetches the length of the dataset 
        
        Args:
            None

        Returns:
            int: Returns the length of the dataset
        '''
        return np.prod(self.shape)


class ImageWiseDataset(Dataset):
    '''
    Dataset class for input to the image-wise network

    An object of this class loads the output of patch-wise network along with it's label for each image from the given directory.
    Used for image-wise network training.

    Attributes:
        path (str): Path of the train/test/val directory
        stride (int): Stride used to extract patches
        labels (dict): A dictionary with keys as filepaths to images and values as associated labels
        names (list): A list containing filepath of all images in an alphabetical order
        shape (tuple): A tuple describing shape of the dataset after all augmentations
        augment_size (int): Denotes the number of augmented images added to the dataset correponding to a single example
    '''

    def __init__(
            self,
            path,
            stride=PATCH_SIZE,
            rotate=False,
            flip=False,
            enhance=False):
        ''' Initialises the class attributes 
        
        Args:
            path (str): Path of the train/test/val directory
            stride (int): Stride used to extract patches (defaults to `ICIAR2018.src.config.PATCH_SIZE` in `ICIAR2018.src.config` Module)
            rotate (bool): A boolean indicating whether to use rotation for augmentation or not
            flip (bool): A boolean indicating whether to use flipping for augmentation or not
            enhance (bool): A boolean indicating whether to use color enhancement for augmentation or not
        
        '''
        super().__init__()

        labels = {
            name: index for index in range(
                len(LABELS)) for name in glob.glob(
                path +
                '/' +
                LABELS[index] +
                '/*.JPG')}

        self.path = path
        self.stride = stride
        self.labels = labels
        self.names = list(sorted(labels.keys()))
        # (files, x_patches, y_patches, rotations, flip, enhance)
        self.shape = (
            len(labels),
            (4 if rotate else 1),
            (2 if flip else 1),
            (2 if enhance else 1))
        self.augment_size = np.prod(self.shape) / len(labels)

    def __getitem__(self, index):
        ''' Fetches an example of given index from the dataset 
        
        Args:
            index (int): Index of the image in the dataset

        Returns:
            tensor: A PyTorch Tensor containing extracted patches from the image
            int: An integer denoting the associated class/label
        '''
        im, rotation, flip, enhance = np.unravel_index(index, self.shape)

        with Image.open(self.names[im]) as img:

            if flip != 0:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if rotation != 0:
                img = img.rotate(rotation * 90)

            if enhance != 0:
                factors = np.random.uniform(.5, 1.5, 3)
                img = ImageEnhance.Color(img).enhance(factors[0])
                img = ImageEnhance.Contrast(img).enhance(factors[1])
                img = ImageEnhance.Brightness(img).enhance(factors[2])

            extractor = PatchExtractor(
                img=img, patch_size=PATCH_SIZE, stride=self.stride)
            patches = extractor.extract_patches()

            label = self.labels[self.names[im]]

            b = torch.zeros((len(patches), 3, PATCH_SIZE, PATCH_SIZE))
            for i in range(len(patches)):
                b[i] = transforms.ToTensor()(patches[i])

            return b, label

    def __len__(self):
        ''' Fetches the length of the dataset 
        
        Args:
            None

        Returns:
            int: Returns the length of the dataset
        '''
        return np.prod(self.shape)


class LabelledDataset(Dataset):
    '''
    Dataset class for validation

    An object of this class loads the extracted patches and the label for each image from the given directory.
    Used for validation functions.

    Attributes:
        path (str): Path of the val directory
        labels (dict): A dictionary with keys as filepaths to images and values as associated labels
        names (list): A list containing filepath of all images in an alphabetical order

    Note:
        The stride used to extracted patches is set to `ICIAR2018.src.config.PATCH_SIZE` in `ICIAR2018.src.config` Module
    '''

    def __init__(self,path):
        ''' Initialises the class attributes 
        
        Args:
            path (str): Path of the val directory        
        '''

        labels = {
            name: index for index in range(
                len(LABELS)) for name in glob.glob(
                path +
                '/' +
                LABELS[index] +
                '/*.JPG')}

        self.labels = labels
        self.names = list(sorted(labels.keys()))
    
    def __getitem__(self,index):
        ''' Fetches an example of given index from the dataset 
        
        Args:
            index (int): Index of the image in the dataset

        Returns:
            tensor: A PyTorch Tensor containing extracted patches from the image
            int: An integer denoting the associated class
            string: Location of the associated file
        '''

        with Image.open(self.names[index]) as img:

            extractor = PatchExtractor(img=img,patch_size=PATCH_SIZE,stride=PATCH_SIZE)
            patches = extractor.extract_patches()

            label = self.labels[self.names[index]]

            b = torch.zeros((len(patches), 3, PATCH_SIZE, PATCH_SIZE))
            for i in range(len(patches)):
                b[i] = transforms.ToTensor()(patches[i])

            return b, label, self.names[index]

    def __len__(self):
        ''' Fetches the length of the dataset 
        
        Args:
            None

        Returns:
            int: Returns the length of the dataset
        '''
        return len(self.names)

class TestDataset(Dataset):
    '''
    Dataset class for test (predictions)

    An object of this class loads the extracted patches for each image from the given directory.
    Used for making predictions (test-time)

    Attributes:
        path (str): Path of the test directory
        names (list): A list containing filepath of all images

    Note:
        The stride used to extracted patches is set to `ICIAR2018.src.config.PATCH_SIZE` in `ICIAR2018.src.config` Module
    '''
    def __init__(self, path):
        ''' Initialises the class attributes 
        
        Args:
            path (str): Path of the test directory        
        '''

        super().__init__()

        if os.path.isdir(path):
            names = [str(name) for name in Path(path).glob('**/*.JPG')]
        else:
            names = [path]

        self.path = path
        self.names = list((names))

    def __getitem__(self, index):
        ''' Fetches an example of given index from the dataset 
        
        Args:
            index (int): Index of the image in the dataset

        Returns:
            tensor: A PyTorch Tensor containing extracted patches from the image
            string: Location of the associated file
        '''

        with Image.open(self.names[index]) as img:

            extractor = PatchExtractor(img=img,patch_size=PATCH_SIZE,stride=PATCH_SIZE)
            patches = extractor.extract_patches()

            b = torch.zeros((len(patches), 3, PATCH_SIZE, PATCH_SIZE))
            for i in range(len(patches)):
                b[i] = transforms.ToTensor()(patches[i])

            return b, self.names[index]

    def __len__(self):
        ''' Fetches the length of the dataset 
        
        Args:
            None

        Returns:
            int: Returns the length of the dataset
        '''
        return len(self.names)