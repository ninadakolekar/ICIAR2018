# -*- coding: utf-8 -*-
"""Defines model functions for PW and IW Network
"""

import time
import ntpath
import datetime
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# PyTorch PACKAGES

import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as ply
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

# sklearn PACKAGES

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

# SOURCE PACKAGES

from .datasets import PatchWiseDataset, ImageWiseDataset, TestDataset
from .config import LABELS, PATCH_SIZE

TRAIN_PATH = '/train'
VALIDATION_PATH = '/test'

mpl.use('Agg')


class BaseModel:
    '''
    Base Model class for initialization and saving of model weights

    An object of this class loads and saves model weights for PW and IW network.

    Attributes:
        args (dict): Arguments defining training/validation/test options
        network (int): Network object (PW Network/IW Network)
        weights (str): Path to the model weights
    '''

    def __init__(self, args, network, weights_path):
        ''' Initialises the class attributes

        Args:
            args (str): Arguments defining training/validation/test options
            network (int): Network object (PW Network/IW Network)
            weights_path (dict): Path to the model weights

        '''
        self.args = args
        self.weights = weights_path
        self.network = network.cuda() if args.cuda else network
        self.load()

    def load(self):
        ''' Load model weights from the model checkpoint

        Args:
            None

        '''
        try:
            if os.path.exists(self.weights):
                self.network.load_state_dict(torch.load(self.weights))
                print(f'Loaded model: {self.weights}')
            else:
                print(f'Model Path Error: {self.weights}')
        except BaseException:
            print(
                'Failed to load pre-trained network ' +
                self.network.name() +
                ' from ' +
                self.weights)

    def save(self):
        ''' Save model weights to the checkpoint directory

        Args:
            None

        '''
        print('Saving model to "{}"'.format(self.weights))
        torch.save(self.network.state_dict(), self.weights)


class PatchWiseModel(BaseModel):
    '''
    Model class for Patchwise Network. Inherits from the Base Model class.

    An object of this class trains (or validates/tests) an instance of (trained) PatchWiseNetwork.

    Attributes:
        args (dict): Arguments defining training/validation/test options
        network (int): Network object (PW Network/IW Network)
        train_loader (torch.utils.data.DataLoader): Dataloader object for the Patchwise network training
        id (str): Training ID
    '''

    def __init__(self, args, network, train=True):
        ''' Initialises the class attributes

        Args:
            args (dict): Arguments defining training/validation/test options
            network (src.networks.PatchWiseNetwork): Network object (PW Network/IW Network)
            train (bool): Boolean indicating whether PatchWise Network is to be trained or not
        '''

        print(f"PatchWiseModel: {network.name()}")
        super(
            PatchWiseModel,
            self).__init__(
            args,
            network,
            os.path.join(
                args.checkpoints_path,
                "weights_" +
                network.name() +
                ".pth"))

        if train:
            self.train_loader = DataLoader(
                dataset=PatchWiseDataset(
                    path=self.args.dataset_path +
                    TRAIN_PATH,
                    stride=self.args.patch_stride,
                    rotate=True,
                    flip=True,
                    enhance=True),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=4)

            if os.path.exists(
                os.path.join(
                    args.checkpoints_path,
                    f"logs_{args.tid}.csv")):
                print(f"Train logs exist")
                exit(0)

        self.id = args.tid

    def train(self):
        ''' Trains the PatchWise Network instance on the given dataset

        Args:
            None

        Returns:
            None
        '''

        logs = pd.DataFrame(
            columns=[
                'epoch',
                'train_loss',
                'train_acc',
                'val_loss',
                'val_acc'])

        self.network.train()
        print('Start training patch-wise network: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))

        optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.args.lr,
            betas=(
                self.args.beta1,
                self.args.beta2))
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.1)
        best_val_acc = 0
        mean_val_acc = 0
        best_epoch = 0
        best_cm = None
        epoch = 0

        for epoch in range(1, self.args.epochs + 1):

            self.network.train()
            scheduler.step()
            stime = datetime.datetime.now()

            correct = 0
            total = 0
            train_loss = 0

            for index, (images, labels) in enumerate(self.train_loader):

                if self.args.cuda:
                    images, labels = images.cuda(), labels.cuda()

                optimizer.zero_grad()
                output = self.network(Variable(images))
                loss = F.nll_loss(output, Variable(labels))
                train_loss += loss
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(output.data, 1)
                correct += torch.sum(predicted == labels)
                total += len(images)

                if index > 0 and index % self.args.log_interval == 0:
                    print('Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                        epoch,
                        self.args.epochs,
                        index * len(images),
                        len(self.train_loader.dataset),
                        100. * index / len(self.train_loader),
                        loss.item(),
                        100 * correct / total
                    ))

            train_loss /= len(self.train_loader.dataset)
            train_acc = 100 * correct / total

            val_loss, val_acc = self.validate(verbose=False)

            if (epoch - 1) % 5 == 0 or epoch == self.args.epochs:
                print(
                    'Saving model to "{}"'.format(
                        self.args.checkpoints_path +
                        '/weights_' +
                        self.network.name() +
                        '_epoch' +
                        str(epoch) +
                        '.pth'))
                torch.save(
                    self.network.state_dict(),
                    self.args.checkpoints_path +
                    '/weights_' +
                    self.network.name() +
                    '_epoch' +
                    str(epoch) +
                    '.pth')

            print('\nEnd of epoch {}, time: {}, val_acc: {}'.format(
                epoch, datetime.datetime.now() - stime, val_acc))

            mean_val_acc += val_acc

            if val_acc > best_val_acc:
                best_epoch = epoch
                best_val_acc = val_acc
                best_cm = confusion_matrix(
                    labels.cpu().numpy(), predicted.cpu().numpy())
                self.save()

            logs.loc[epoch] = [epoch, train_loss, train_acc, val_loss, val_acc]
            logs.to_csv(
                os.path.join(
                    self.args.checkpoints_path,
                    f"logs_{self.id}.csv"),
                index=False)

        print(
            '\nEnd of training, best accuracy: {}, best_epoch: {},mean accuracy: {}\n'.format(
                best_val_acc,
                best_epoch,
                mean_val_acc //
                epoch))

    def validate(self, verbose=True):
        ''' Validates the trained/partially-trained PatchWise Network instance on the given dataset only

        Args:
            verbose (bool): If true, validation statistics will be printed to STDOUT

        Returns:
            Validation Loss (float)
            Validation Accuracy (float)
        '''
        self.network.eval()

        test_loss = 0

        correct = 0
        classes = len(LABELS)

        tp = [0] * classes
        tpfp = [0] * classes
        tpfn = [0] * classes
        precision = [0] * classes
        recall = [0] * classes
        f1 = [0] * classes

        test_loader = DataLoader(
            dataset=PatchWiseDataset(
                path=self.args.dataset_path +
                VALIDATION_PATH,
                stride=self.args.patch_stride),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4)
        if verbose:
            print('\nEvaluating....')

        for images, labels in test_loader:

            if self.args.cuda:
                images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                output = self.network(Variable(images))

            test_loss += F.nll_loss(output,
                                    Variable(labels),
                                    reduction='sum').item()
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(predicted == labels)

            for label in range(classes):
                t_labels = labels == label
                p_labels = predicted == label
                tp[label] += torch.sum(t_labels == (p_labels * 2 - 1))
                tpfp[label] += torch.sum(p_labels)
                tpfn[label] += torch.sum(t_labels)

        for label in range(classes):
            precision[label] += (tp[label] / (tpfp[label] + 1e-8))
            recall[label] += (tp[label] / (tpfn[label] + 1e-8))
            f1[label] = 2 * precision[label] * recall[label] / \
                (precision[label] + recall[label] + 1e-8)

        test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / len(test_loader.dataset)

        if verbose:
            print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)
            ))

            for label in range(classes):
                print('{}:  \t Precision: {:.2f},  Recall: {:.2f},  F1: {:.2f}'.format(
                    LABELS[label], precision[label], recall[label], f1[label]))

            print('')

        return test_loss, test_acc

    def test(self, path, verbose=True):
        ''' Computes the predictions of the morphology(ies) for given dataset (location) using the trained PatchWise Network instance only

        Args:
            path (str): Location of the dataset on which predictions are to be computed
            verbose: If true, validation statistics will be printed to STDOUT

        Returns:
            List of predicted morphology for each instance in the dataset (list)
        '''

        self.network.eval()
        dataset = TestDataset(path=path, stride=PATCH_SIZE, augment=False)
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        stime = datetime.datetime.now()

        if verbose:
            print('\t sum\t\t max\t\t maj\t')

        res = []

        for index, (image, file_name) in enumerate(data_loader):
            image = image.squeeze()
            if self.args.cuda:
                image = image.cuda()

            output = self.network(Variable(image))
            _, predicted = torch.max(output.data, 1)

            sum_prob = 2 - \
                np.argmax(np.sum(np.exp(output.data.cpu().numpy()), axis=0)[::-1])
            max_prob = 2 - \
                np.argmax(np.max(np.exp(output.data.cpu().numpy()), axis=0)[::-1])
            maj_prob = 2 - \
                np.argmax(np.sum(np.eye(3)[np.array(predicted.cpu().numpy()).reshape(-1)], axis=0)[::-1])

            res.append([sum_prob, max_prob, maj_prob, file_name[0]])
            if verbose:
                np.sum(output.data.cpu().numpy(), axis=0)
                print('{}) \t {} \t {} \t {} \t {}'.format(
                    str(index + 1).rjust(2, '0'),
                    LABELS[sum_prob].ljust(8),
                    LABELS[max_prob].ljust(8),
                    LABELS[maj_prob].ljust(8),
                    ntpath.basename(file_name[0])))

        if verbose:
            print(
                '\nInference time: {}\n'.format(
                    datetime.datetime.now() -
                    stime))

        return res

    def output(self, input_tensor):
        ''' Computes the feature-map for an input image using the trained PatchWise Network instance

        Args:
            input_tensor (torch.Tensor): Input cell-line image

        Returns:
            Feature-map corresponding to the input image tensor (torch.Tensor)
        '''
        self.network.eval()
        with torch.no_grad():
            res = self.network.features(Variable(input_tensor))
        return res.squeeze()


class ImageWiseModel(BaseModel):
    '''
    Model class for ImageWise Network. Inherits from the Base Model class.

    An object of this class trains (or validates/tests) an instance of (trained) ImageWise Network.

    Attributes:
        args (dict): Arguments defining training/validation/test options
        patch_wise_model (src.networks.PatchWiseNetwork): The trained Patch-wise model to be used while training/validating Image-wise network
        _test_loader (torch.utils.data.DataLoader): Dataloader object for the Patchwise network testing time
    '''

    def __init__(self, args, image_wise_network, patch_wise_network):
        ''' Initialises the class attributes

        Args:
            args (dict): Arguments defining training/validation/test options
            image_wise_network (src.networks.ImageWiseNetwork): Network instance (PW Network/IW Network)
            patch_wise_network (src.networks.PatchWiseNetwork): The trained Patch-wise model to be used while training/validating Image-wise network
        '''

        print(f"ImageWiseModel: {image_wise_network.name()}")
        super(
            ImageWiseModel,
            self).__init__(
            args,
            image_wise_network,
            args.checkpoints_path +
            '/weights_' +
            image_wise_network.name() +
            '.pth')

        self.patch_wise_model = PatchWiseModel(
            args, patch_wise_network, train=False)
        self._test_loader = None

    def train(self):
        ''' Trains the ImageWise Network instance on the given dataset

        Args:
            None

        Returns:
            None
        '''

        self.network.train()
        print('ImageWiseModel.train: Evaluating patch-wise model...')

        train_loader = self._patch_loader(
            self.args.dataset_path + TRAIN_PATH, True)

        print(
            'ImageWiseModel.train: Start training image-wise network: {}\n'.format(
                time.strftime('%Y/%m/%d %H:%M')))

        optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.args.lr,
            betas=(
                self.args.beta1,
                self.args.beta2))
        best = 0
        mean = 0
        epoch = 0

        for epoch in range(1, self.args.epochs + 1):

            print(f"Starting epoch {epoch}...")

            self.network.train()
            stime = datetime.datetime.now()

            correct = 0
            total = 0

            for index, (images, labels) in enumerate(train_loader):

                if self.args.cuda:
                    images, labels = images.cuda(), labels.cuda()

                optimizer.zero_grad()
                output = self.network(Variable(images))
                loss = F.nll_loss(output, Variable(labels))
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(output.data, 1)
                correct += torch.sum(predicted == labels)
                total += len(images)

                if index > 0 and index % 10 == 0:
                    print('Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                        epoch,
                        self.args.epochs,
                        index * len(images),
                        len(train_loader.dataset),
                        100. * index / len(train_loader),
                        loss.item(),
                        100 * correct / total
                    ))

            print('\nEnd of epoch {}, time: {}'.format(
                epoch, datetime.datetime.now() - stime))

            if (epoch - 1) % 5 == 0 or epoch == self.args.epochs:
                print(
                    'Saving model (periodic) to "{}"'.format(
                        self.args.checkpoints_path +
                        '/weights_' +
                        self.network.name() +
                        '_epoch' +
                        str(epoch) +
                        '.pth'))
                torch.save(
                    self.network.state_dict(),
                    self.args.checkpoints_path +
                    '/weights_' +
                    self.network.name() +
                    '_epoch' +
                    str(epoch) +
                    '.pth')

            acc = self.validate()
            mean += acc
            if acc > best:
                best = acc
                self.save()

        print(
            '\nEnd of training, best accuracy: {}, mean accuracy: {}\n'.format(
                best,
                mean //
                epoch))

    def validate(self, verbose=True, roc=False):
        ''' Validates the trained/partially-trained Imagewise Network instance on the given dataset only

        Args:
            verbose (bool): If set to true, validation statistics will be printed to STDOUT
            roc (bool): If set to true, ROC curves will be generated for each of the classes

        Returns:
            Validation Loss (float)
            Validation Accuracy (float)
        '''

        self.network.eval()

        if self._test_loader is None:
            self._test_loader = self._patch_loader(
                self.args.dataset_path + VALIDATION_PATH, False)

        val_loss = 0
        correct = 0
        classes = len(LABELS)

        tp = [0] * classes
        tpfp = [0] * classes
        tpfn = [0] * classes
        precision = [0] * classes
        recall = [0] * classes
        f1 = [0] * classes

        if verbose:
            print('\nEvaluating....')

        labels_true = []
        labels_pred = np.empty((0, 3))

        for images, labels in self._test_loader:

            if self.args.cuda:
                images, labels = images.cuda(), labels.cuda()

            with torch.no_grad():
                output = self.network(Variable(images))

            val_loss += F.nll_loss(output, Variable(labels),
                                   reduction='sum').item()
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(predicted == labels)

            labels_true = np.append(labels_true, labels.cpu().numpy())
            labels_pred = np.append(
                labels_pred, torch.exp(
                    output.data).cpu().numpy(), axis=0)

            for label in range(classes):
                t_labels = labels == label
                p_labels = predicted == label
                tp[label] += torch.sum(t_labels == (p_labels * 2 - 1))
                tpfp[label] += torch.sum(p_labels)
                tpfn[label] += torch.sum(t_labels)

        for label in range(classes):
            precision[label] += (tp[label] / (tpfp[label] + 1e-8))
            recall[label] += (tp[label] / (tpfn[label] + 1e-8))
            f1[label] = 2 * precision[label] * recall[label] / \
                (precision[label] + recall[label] + 1e-8)

        val_loss /= len(self._test_loader.dataset)
        acc = 100. * correct / len(self._test_loader.dataset)

        print(f"Accuracy: {acc}")

        if roc == 1:
            labels_true = label_binarize(labels_true, classes=range(classes))
            for lbl in range(classes):
                fpr, tpr, _ = roc_curve(
                    labels_true[:, lbl], labels_pred[:, lbl])
                roc_auc = auc(fpr, tpr)
                plt.plot(
                    fpr, tpr, lw=2, label='{} (AUC: {:.1f})'.format(
                        LABELS[lbl], roc_auc * 100))

            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.title('Receiver Operating Characteristic')
            plt.show()

        if verbose:
            print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                val_loss,
                correct,
                len(self._test_loader.dataset),
                acc
            ))

            for label in range(classes):
                print('{}:  \t Precision: {:.2f},  Recall: {:.2f},  F1: {:.2f}'.format(
                    LABELS[label], precision[label], recall[label], f1[label]))

            print('')

        return acc

    def test(self, path, verbose=True, ensemble=True):
        ''' Computes the predictions of the morphology(ies) for given dataset (location) using the trained ImageWise Network

        Args:
            path (str): Location of the dataset on which predictions are to be computed
            verbose: If true, validation statistics will be printed to STDOUT

        Returns:
            List of predicted morphology for each instance in the dataset (list)
        '''

        self.network.eval()
        dataset = TestDataset(path=path, stride=PATCH_SIZE, augment=ensemble)
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        stime = datetime.datetime.now()

        if verbose:
            print('')

        res = []

        for index, (image, file_name) in enumerate(data_loader):
            n_bins, n_patches = image.shape[1], image.shape[2]
            image = image.view(-1, 3, PATCH_SIZE, PATCH_SIZE)

            if self.args.cuda:
                image = image.cuda()

            patches = self.patch_wise_model.output(image)
            patches = patches.view(n_bins, -1, 64, 64)

            if self.args.cuda:
                patches = patches.cuda()

            output = self.network(patches)
            _, predicted = torch.max(output.data, 1)

            output = output.detach().cpu().numpy()
            predicted = predicted.cpu().numpy()

            # maj_prop: majority voting: create a one-hot vector of predicted values: (12, 4),
            # sum among y axis: (1, 4), reverse, and take the index  of the
            # largest value

            maj_prob = 2 - \
                np.argmax(np.sum(np.eye(3)[np.array(predicted).reshape(-1)], axis=0)[::-1])

            confidence = np.sum(np.array(predicted) == maj_prob) / \
                n_bins if ensemble else torch.max(torch.exp(output.data))
            confidence = np.round(confidence * 100, 2)

            res.append([maj_prob, confidence, file_name[0]])

            if verbose:
                print('{}) {} ({}%) \t {}'.format(
                    str(index).rjust(2, '0'),
                    LABELS[maj_prob],
                    confidence,
                    ntpath.basename(file_name[0])))

        if verbose:
            print(
                '\nInference time: {}\n'.format(
                    datetime.datetime.now() -
                    stime))

        return res

    def _patch_loader(self, path, augment):
        ''' Private-scope function to load the PW Network feature maps for input to the ImageWise Network

        Args:
            path (str): Dataset location
            augment: If set to true, the patches will be augmented (rotation,flip,color-augmentation)

        Returns:
            List of predicted morphology for each instance in the dataset (list)
        '''

        images_path = '{}/{}_images.npy'.format(
            self.args.checkpoints_path, self.network.name())
        labels_path = '{}/{}_labels.npy'.format(
            self.args.checkpoints_path, self.network.name())

        if self.args.debug and augment and os.path.exists(images_path):
            np_images = np.load(images_path)
            np_labels = np.load(labels_path)

        else:
            dataset = ImageWiseDataset(
                path=path,
                stride=PATCH_SIZE,
                flip=augment,
                rotate=augment,
                enhance=augment)

            bsize = 8
            output_loader = DataLoader(
                dataset=dataset,
                batch_size=bsize,
                shuffle=True,
                num_workers=4)
            output_images = []
            output_labels = []

            for index, (images, labels) in enumerate(output_loader):
                if index > 0 and index % 10 == 0:
                    print('{} images loaded'.format(
                        int((index * bsize) / dataset.augment_size)))

                if self.args.cuda:
                    images = images.cuda()

                bsize = images.shape[0]

                res = self.patch_wise_model.output(
                    images.view((-1, 3, 512, 512)))
                res = res.view((bsize, -1, 64, 64)).data.cpu().numpy()

                for i in range(bsize):
                    output_images.append(res[i])
                    output_labels.append(labels.numpy()[i])

            np_images = np.array(output_images)
            np_labels = np.array(output_labels)

            if self.args.debug and augment:
                np.save(images_path, np_images)
                np.save(labels_path, np_labels)

        images, labels = torch.from_numpy(
            np_images), torch.from_numpy(np_labels).squeeze()

        return DataLoader(
            dataset=TensorDataset(images, labels),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2
        )
