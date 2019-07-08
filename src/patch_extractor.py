# -*- coding: utf-8 -*-
"""Defines class for patch extraction
"""


class PatchExtractor(object):
    '''
    Defines parameters and functions related to PatchExtractor

    An object of this class loads the extracted patches and the label for each image from the given directory.
    Used for patch-wise network training.

    Attributes:
        img (:py:class:`~PIL.Image.Image`): Image from which patches need to be extracted
        stride (int): Stride used to extract patches
        size (int): Size of the patch
    '''

    def __init__(self, img, patch_size, stride):
        '''
        Initialises the class attributes

        Args:
            img (:py:class:`~PIL.Image.Image`): Image from which patches need to be extracted
            stride (int): Stride used to extract patches
            size (int): Size of the patch
        '''
        self.img = img
        self.size = patch_size
        self.stride = stride

    def extract_patches(self):
        '''
        Extracts all patches from input image

        Args:
            None

        Returns:
            list: A list of extracted patches (:py:class:`~PIL.Image.Image` objects)
        '''
        wp, hp = self.shape()
        return [self.extract_patch((w, h))
                for h in range(hp) for w in range(wp)]

    def extract_patch(self, patch):
        '''
        Extracts a specific patch from input image

        Args:
            tuple: A tuple of integers

        Returns:
            list: Extracted patch (:py:class:`~PIL.Image.Image` object)
        '''
        return self.img.crop((
            patch[0] * self.stride,  # left
            patch[1] * self.stride,  # up
            patch[0] * self.stride + self.size,  # right
            patch[1] * self.stride + self.size  # down
        ))

    def shape(self):
        '''
        Computes number of patches along the width and height

        Args:
            None

        Returns:
            int: Number of patches along the width
            int: Number of patches along the height
        '''
        wp = int((self.img.width - self.size) / self.stride + 1)
        hp = int((self.img.height - self.size) / self.stride + 1)
        return wp, hp
