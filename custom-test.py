import matplotlib
matplotlib.use('Agg')

import os
import glob
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch.autograd import Variable

# SOURCE PACKAGES #
from src.networks import PatchWiseNetwork, ImageWiseNetwork
from src.patch_extractor import PatchExtractor

# CONSTANTS
LABELS = ['grape', 'round', 'stellate']
PATCH_SIZE = 512

class LabelledDataset(Dataset):

    def __init__(self,path):

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

class TestDataset(Dataset):
    def __init__(self, path):
        super().__init__()

        if os.path.isdir(path):
            names = [str(name) for name in Path(path).glob('**/*.JPG')]
        else:
            names = [path]

        self.path = path
        self.names = list((names))

    def __getitem__(self, index):

        with Image.open(self.names[index]) as img:

            extractor = PatchExtractor(img=img,patch_size=PATCH_SIZE,stride=PATCH_SIZE)
            patches = extractor.extract_patches()

            b = torch.zeros((len(patches), 3, PATCH_SIZE, PATCH_SIZE))
            for i in range(len(patches)):
                b[i] = transforms.ToTensor()(patches[i])

            return b, self.names[index]

    def __len__(self):
        return len(self.names)



class ModelArgs:

    def __init__(self):

        parser = argparse.ArgumentParser(description='Classification of morphology')

        parser.add_argument('--testset-path',type=str,required='True',help='Path to test directory or file')
        parser.add_argument('--val',action='store_true',default=False,help='Set mode to validation')
        parser.add_argument('--pw-checkpoints-path',type=str,required=True,help='Path to saved PW model checkpoints')
        parser.add_argument('--iw-checkpoints-path',type=str,required=True,help='Path to saved IW model checkpoints')
        parser.add_argument('--no-cuda',action='store_true',default=False,help='Disables CUDA evaluation')
        parser.add_argument('--seed',type=int,default=42,help='Random seed (default: 42)')
        parser.add_argument('--outdir',required=True,help='Directory to output CSV and ROC')
        parser.add_argument('--verbose',action='store_true',default=False,help='Enables verbose evaluation')
        parser.add_argument('--gpu-ids',type=str,default='0',help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--channels',type=int,default=1,help='number of channels created by the patch-wise network that feeds into the image-wise network (default: 1)')
        self._parser = parser

    def parse(self):

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


if __name__ == "__main__":

    args = ModelArgs().parse()

    def verbose(msg):
        if args.verbose:
            print(f"INFO: {msg}")

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    pw_network = PatchWiseNetwork(args.channels,init=False)
    pw_network = pw_network.cuda() if args.cuda else pw_network
    verbose(f"Loaded PW Network: {pw_network.name()}")

    iw_network = ImageWiseNetwork(args.channels,init=False)
    iw_network = iw_network.cuda() if args.cuda else iw_network
    verbose(f"Loaded IW Network: {iw_network.name()}")

    pw_checkpoint = args.pw_checkpoints_path
    iw_checkpoint = args.iw_checkpoints_path

    assert(os.path.exists(pw_checkpoint) and os.path.isfile(pw_checkpoint))
    assert(os.path.exists(iw_checkpoint) and os.path.isfile(pw_checkpoint))

    pw_network.load_state_dict(torch.load(pw_checkpoint))
    verbose("Loaded PW Weights: {pw_checkpoint}")

    iw_network.load_state_dict(torch.load(iw_checkpoint))
    verbose("Loaded IW Weights: {iw_checkpoint}")

    pw_network.eval()
    iw_network.eval()

    # Load Dataset
    if args.val:
        verbose("Evaluating samples")

        resluts_df = pd.DataFrame(columns=["Filepath","True Label","Predicted","Confidence"])

        CellsDataset = LabelledDataset(path=args.testset_path)
        CellsLoader = DataLoader(dataset=CellsDataset,batch_size=1,shuffle=True,num_workers=4)

        verbose("Dataset and DataLoader (labelled) objects created")

        labels_true = []
        labels_pred = np.empty((0, 3))

        classes = len(LABELS)
        tp = [0] * classes
        tpfp = [0] * classes
        tpfn = [0] * classes
        precision = [0] * classes
        recall = [0] * classes
        f1 = [0] * classes

        correct = 0
        total = 0

        for index, (images,label,filepath) in enumerate(CellsLoader):

            if total%10 == 0 and total !=0:
                verbose(f"Evaluated {total} images")

            pw_network.eval()
            with torch.no_grad():
                images = images.view((-1,3,512,512))
                if args.cuda:
                    images = images.cuda()
                pw_output = pw_network.features(Variable(images))
            pw_output = pw_output.squeeze().view((1, -1, 64, 64)).data.cpu()

            iw_network.eval()
            with torch.no_grad():
                if args.cuda:
                    pw_output = pw_output.cuda()
                iw_output = iw_network(pw_output)
            
            _, predicted = torch.max(iw_output.data,1)

            # ROC Logging
            labels_true = np.append(labels_true, label)
            labels_pred = np.append(
                labels_pred, torch.exp(iw_output.data).cpu().numpy(), axis=0)

            iw_output = iw_output.detach().cpu()
            predicted = predicted.cpu()

            for idx in range(classes):
                t_labels = label == idx
                p_labels = predicted == idx
                tp[idx] += torch.sum(t_labels == (p_labels * 2 - 1))
                tpfp[idx] += torch.sum(p_labels)
                tpfn[idx] += torch.sum(t_labels)

            iw_output = iw_output.numpy()
            predicted = predicted.numpy()

            maj_prob = 2 - np.argmax(np.sum(np.eye(3)[np.array(predicted).reshape(-1)], axis=0)[::-1])

            confidence = np.sum(np.array(predicted) == maj_prob)
            confidence = np.round(confidence * 100, 2)

            total += 1
            if maj_prob == label:
                correct += 1

            resluts_df.loc[index] = [filepath,LABELS[label],LABELS[maj_prob],confidence]

        verbose("Evaluation completed")
        verbose(f"Accuracy: {correct/total}")

        verbose("Computing ROC...")

        for label in range(classes):
            precision[label] += (tp[label] / (tpfp[label] + 1e-8))
            recall[label] += (tp[label] / (tpfn[label] + 1e-8))
            f1[label] = 2 * precision[label] * recall[label] / \
                (precision[label] + recall[label] + 1e-8)

        labels_true = label_binarize(labels_true, classes=range(classes))

        for lbl in range(classes):
            fpr, tpr, _ = roc_curve(labels_true[:, lbl], labels_pred[:, lbl])
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
            plt.savefig(os.path.join(args.outdir,f"ROC_{LABELS[lbl]}.png"))

        verbose(f"ROC curves saved to {args.outdir}")

        resluts_df.to_csv(os.path.join(args.outdir,"results.csv"),index=False)
        verbose(f"Results saved to {os.path.join(args.outdir,'results.csv')}")

    else:
        verbose("Computing predictions")

        resluts_df = pd.DataFrame(columns=["Filepath","Predicted","Confidence"])

        CellsDataset = TestDataset(path=args.testset_path)
        CellsLoader = DataLoader(dataset=CellsDataset,batch_size=1,shuffle=True,num_workers=4)

        verbose("Dataset and DataLoader objects created")

        for index, (images,filepath) in enumerate(CellsLoader):

            if index%10 == 0 and index !=0:
                verbose(f"Evaluated {index} images")

            pw_network.eval()
            with torch.no_grad():
                images = images.view((-1,3,512,512))
                if args.cuda:
                    images = images.cuda()
                pw_output = pw_network.features(Variable(images))
            pw_output = pw_output.squeeze().view((1, -1, 64, 64)).data.cpu()

            iw_network.eval()
            with torch.no_grad():
                if args.cuda:
                    pw_output = pw_output.cuda()
                iw_output = iw_network(pw_output)
            
            _, predicted = torch.max(iw_output.data,1)

            iw_output = iw_output.detach().cpu().numpy()
            predicted = predicted.cpu().numpy()

            maj_prob = 2 - np.argmax(np.sum(np.eye(3)[np.array(predicted).reshape(-1)], axis=0)[::-1])

            confidence = np.sum(np.array(predicted) == maj_prob)
            confidence = np.round(confidence * 100, 2)

            resluts_df.loc[index] = [filepath,LABELS[maj_prob],confidence]

        verbose("Predictions completed")

        resluts_df.to_csv(os.path.join(args.outdir,"results.csv"),index=False)
        verbose(f"Results saved to {os.path.join(args.outdir,'results.csv')}")
