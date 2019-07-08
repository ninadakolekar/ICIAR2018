import matplotlib
matplotlib.use('Agg')

import os
import glob
from pathlib import Path
import argparse
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

# SOURCE PACKAGES #
from src.networks import PatchWiseNetwork, ImageWiseNetwork
from src.patch_extractor import PatchExtractor
from src.config import LABELS, PATCH_SIZE
from src.options import TrainingOptions

class LabelledDataset(Dataset):

    def __init__(self,path):

        df = pd.read_csv(path)

        labels = {df.iloc[i,0].replace("'",'')[1:-1].replace(',','').strip():LABELS.index(df.iloc[i,1].lower()) for i in range(len(df))}

        self.labels = labels
        self.names = list(sorted(labels.keys()))
    
    def __getitem__(self,index):

        with Image.open(self.names[index]) as img:

            extractor = PatchExtractor(img=img,patch_size=PATCH_SIZE,stride=PATCH_SIZE)
            patches = extractor.extract_patches()

            label = self.labels[self.names[index]]

            b = torch.zeros((len(patches), 3, PATCH_SIZE, PATCH_SIZE))
            for i in range(len(patches)):
                b[i] = transforms.ToTensor()(patches[i])

            return b, label, self.names[index]

    def __len__(self):
        return len(self.names)

if __name__ == "__main__":

    args = TrainingOptions().parse()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_dataset = LabelledDataset("/home/nitish/Desktop/ninad/matrigel_train.csv")
    test_dataset = LabelledDataset("/home/nitish/Desktop/ninad/matrigel_val.csv")

    pw_network = PatchWiseNetwork(args.channels,init=False)
    pw_network = pw_network.cuda() if args.cuda else pw_network
    print(f"Loaded PW Network: {pw_network.name()}")

    iw_network = ImageWiseNetwork(args.channels,init=False)
    iw_network = iw_network.cuda() if args.cuda else iw_network
    print(f"Loaded IW Network: {iw_network.name()}")

    assert(os.path.isdir(args.checkpoints_path))

    pw_checkpoint = os.path.join(args.checkpoints_path,"weights_"+pw_network.name()+".pth")
    iw_checkpoint = os.path.join(args.checkpoints_path,"weights_"+iw_network.name()+".pth")

    assert(os.path.exists(args.checkpoints_path))

    assert(os.path.exists(pw_checkpoint) and os.path.isfile(pw_checkpoint))
    assert(os.path.exists(iw_checkpoint) and os.path.isfile(pw_checkpoint))

    pw_network.load_state_dict(torch.load(pw_checkpoint))
    print(f"Loaded PW Weights: {pw_checkpoint}")

    iw_network.load_state_dict(torch.load(iw_checkpoint))
    print(f"Loaded IW Weights: {iw_checkpoint}")

    pw_network = nn.DataParallel(pw_network,device_ids=[0,1])
    iw_network = nn.DataParallel(iw_network,device_ids=[0,1])

    train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4)
    test_loader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=True,num_workers=4)

    # Fine-tune IW Network
    optimizer = optim.Adam(
            iw_network.parameters(),
            lr=args.lr,
            betas=(
                args.beta1,
                args.beta2))
    best = 0
    mean = 0
    epoch = 0

    for epoch in range(1, args.epochs + 1):

        print(f"Starting epoch {epoch}...")

        stime = datetime.datetime.now()

        correct = 0
        total = 0
        train_loss = 0

        for index, (images,label,filepath) in enumerate(train_loader):

            if index%10 == 0 and index !=0:
                print(f"Evaluated {index} images")

            pw_network.eval()
            with torch.no_grad():
                images = images.view((-1,3,512,512))
                if args.cuda:
                    images = images.cuda()
                pw_output = pw_network.module.features(Variable(images))
            pw_output = pw_output.squeeze().view((1, -1, 64, 64)).data.cpu()

            iw_network.train()

            optimizer.zero_grad()
            iw_output = iw_network(Variable(pw_output.cuda())).cpu()
            loss = F.nll_loss(iw_output, Variable(label))
            train_loss += loss
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(iw_output.data, 1)
            correct += torch.sum(predicted == label)
            total += len(images)

            if index > 0 and index % args.log_interval == 0:
                print('Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                    epoch,
                    args.epochs,
                    index * len(images),
                    len(train_loader.dataset),
                    100. * index / len(train_loader),
                    loss.item(),
                    100 * correct / total
                ))

        train_loss /= len(train_loader.dataset)
        train_acc = 100 * correct / total

        correct = 0
        total = 0
        val_loss = 0

        for index, (images,label,filepath) in enumerate(test_loader):

            pw_network.eval()
            with torch.no_grad():
                images = images.view((-1,3,512,512))
                if args.cuda:
                    images = images.cuda()
                pw_output = pw_network.module.features(Variable(images))
            pw_output = pw_output.squeeze().view((1, -1, 64, 64)).data.cpu()

            iw_network.eval()
            with torch.no_grad():
                iw_output = iw_network(Variable(pw_output)).cpu()
                loss = F.nll_loss(iw_output, Variable(label))
                val_loss += loss

            _, predicted = torch.max(iw_output.data, 1)
            correct += torch.sum(predicted == label)
            total += len(images)

        val_loss /= total
        val_acc = 100 * correct/total

        print(
        'Saving model to "{}"'.format(
            args.checkpoints_path +
            '/ft_weights_' +
            iw_network.module.name() +
            '_epoch' +
            str(epoch) +
            '.pth'))
        torch.save(
            iw_network.state_dict(),
            args.checkpoints_path +
            '/ft_weights_' +
            iw_network.module.name() +
            '_epoch' +
            str(epoch) +
            '.pth')

        print('\nEnd of epoch {}, time: {}, val_acc: {}, val_loss: {}'.format(
                epoch, datetime.datetime.now() - stime, val_acc,val_loss))



            



    
