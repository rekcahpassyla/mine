import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

import socket
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_uniform
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler


import loadsplit
import os


class TransformDataset(Dataset):

    def __init__(self, data, labels):

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=10, shear=10),
            transforms.ToTensor()
        ])
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data.shape[0])

    def __getitem__(self, idx):
        item = self.data[idx]
        item = self.transform(item.numpy().astype(np.float32))
        return item.double(), self.labels[idx]


def extract(data, normalise=True):
    labels = data[:, 0].astype(int)
    data = data[:, 1:]
    indices = np.arange(labels.size)
    np.random.shuffle(indices)
    data_selected = data[indices]
    data_selected = data_selected.reshape(-1, 16, 16)
    labels_selected = labels[indices]
    data_selected = transforms.ToTensor()(data_selected)
    data_selected = data_selected.transpose(1, 0)
    labels_selected = torch.tensor(labels_selected)
    if normalise:
        mean = data_selected.mean()
        std = data_selected.std()
        data_selected = transforms.Normalize([mean], [std])(data_selected)
    return data_selected, labels_selected


def get_loaders(batchsize=500, test_frac=0.2):
    train, test, _, _ = loadsplit.randomsample(loadsplit.data, test_frac)
    # now take another 0.1 of the training data as the validation data.
    # this helps us decide when to stop.
    nvalid = int(train.shape[0]*0.1)
    valid = train[nvalid:]
    train = train[:nvalid]
    trainxs, trainys = extract(train)
    testxs, testys = extract(test)
    validxs, validys = extract(valid)
    train_indices = np.arange(len(trainys))
    np.random.shuffle(train_indices)
    test_indices = np.arange(len(testys))
    np.random.shuffle(test_indices)
    valid_indices = np.arange(len(validys))
    np.random.shuffle(valid_indices)
    test_sample = SubsetRandomSampler(test_indices)
    test_dataset = TensorDataset(testxs, testys)
    train_sample = SubsetRandomSampler(train_indices)
    train_dataset = TransformDataset(trainxs, trainys)
    valid_sample = SubsetRandomSampler(valid_indices)
    valid_dataset = TensorDataset(validxs, validys)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, sampler=train_sample, batch_size=batchsize,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, sampler=test_sample, batch_size=batchsize
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, sampler=valid_sample, batch_size=batchsize
    )
    return train_loader, test_loader, valid_loader



# Detect if GPUs are available
GPU = torch.cuda.is_available()

# If you have a problem with your GPU, set this to "cpu" manually
device = torch.device("cuda:0" if GPU else "cpu")
epochs = 200
rootdir = ''

if socket.gethostname() == 'tempoyak':
    device = "cpu"
    epochs = 100
    rootdir = 'tmp'

USE_GPU = device != "cpu"


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.e_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.e_conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.e_conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.e_batchnorm = nn.BatchNorm2d(32)
        self.e_activation = F.relu
        # halve all dimensions with pooling layer with kernel/stride = 2
        self.pool = nn.MaxPool2d(2, 2)

        for layer in [self.e_conv1, self.e_conv2, self.e_conv3]:
            xavier_uniform_(layer.weight)
        self.linear1 = nn.Linear(8*4*4, 64)
        self.linear2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.e_activation(self.e_conv1(x))
        x = self.e_batchnorm(x)
        x = self.pool(x)
        x = self.e_activation(self.e_conv2(x))
        x = self.e_activation(self.e_conv3(x))
        x = self.pool(x)
        # must reshape output of CNN into flat for the final layer
        x = x.view(-1, 8*4*4)
        x = self.dropout(x)
        leaky = torch.nn.LeakyReLU()
        x = leaky(self.linear1(x))
        x = leaky(self.linear2(x))

        return x


    @classmethod
    def from_saved(cls, filename, map_location=None):
        model = cls()
        kws = {}
        if map_location is not None:
            kws['map_location'] = map_location
        model.load_state_dict(torch.load(filename, **kws))
        return model

    def do_train(self, train_loader, loss_function, optimiser):
        # not called train() because that means something else in the superclass
        train_loss = 0.0
        self.train()

        Ntotal = len(train_loader.sampler)
        for data, label in train_loader:
            #if len(data.shape) == 3:
            #data = data.unsqueeze(1)
            if USE_GPU:
                data = data.to(device)
                label = label.to(device)
            optimiser.zero_grad()
            output = self(data)

            loss = loss_function(output, label)
            loss.backward()
            optimiser.step()
            thistrain = loss.item() * data.size(0)
            train_loss += thistrain

        last_loss = None
        min_loss = np.Inf
        train_loss = train_loss / Ntotal
        return train_loss

    def validate(self, valid_loader, loss_function):
        # run validation during training, return relevant statistics
        # valid_loader: data loader containing validation set
        class_correct = np.zeros(10)
        class_total = np.zeros(10)
        # go into evaluation mode
        self.eval()
        valid_loss = 0.0
        for data, label in valid_loader:
            data = data.unsqueeze(1)
            if USE_GPU:
                data = data.to(device)
                label = label.to(device)
            output = self(data)
            loss = loss_function(output, label)
            thisvalid = loss.item() * data.size(0)
            valid_loss += thisvalid
            _, pred = torch.max(output, 1)
            correct = pred.eq(label.data)

            for i in range(len(label)):
                clabel = label.data[i]
                class_correct[clabel] += correct[i].item()
                class_total[clabel] += 1
        valid_loss = valid_loss / len(valid_loader.sampler)
        return class_correct, class_total, valid_loss

    def test(self, test_loader, loss_function, tag):
        class_correct = np.zeros(10)
        class_total = np.zeros(10)
        test_loss = 0.0
        self.eval()
        for data, target in test_loader:
            data = data.unsqueeze(1)
            if USE_GPU:
                # This transfers your data to the GPU
                # If you don't do this, you'll get an error
                data = data.to(device)
                target = target.to(device)
            # predict
            output = self(data)
            loss = loss_function(output, target)
            test_loss += loss.item()*data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = pred.eq(target.data)
            # calculate test accuracy for each object class
            for i in range(len(target)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
        test_loss = test_loss/len(test_loader.sampler)
        print(f'Test Loss: {test_loss:.6f}\n')
        for i in range(10):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    str(i), 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print(f'Test Accuracy of {i}: N/A (no training examples)')
        print('\n%s\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            tag,
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
        return class_correct, class_total, test_loss



def do_run(args):
    split, ceta, batchsize = args
    tag = f"batchsize={batchsize}_eta={ceta}"
    ctag = f"{tag}_{split}"
    print(f"Running: {ctag}")
    classifier = CNNClassifier()
    # this is what makes the split
    train_loader, test_loader, valid_loader = get_loaders(batchsize=batchsize)
    closs = nn.CrossEntropyLoss()
    subdir = f'{rootdir}cnn_results_augmented/{tag}'
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    cfilename = f'{subdir}/cnn_{ctag}'
    copt = optim.AdamW(classifier.parameters(), lr=ceta)

    if USE_GPU:
        classifier = classifier.to(device)

    classifier = classifier.double()

    valid_loss_min = np.Inf

    with open(f"{cfilename}.csv", 'w') as fh:
        fh.write('Epoch,train_loss,validation_loss,0,1,2,3,4,5,6,7,8,9\n')
        fh.flush()
        for epoch in range(epochs):
            train_loss = classifier.do_train(train_loader, closs, copt)
            class_correct, class_total, valid_loss = classifier.validate(
                valid_loader, closs)

            stats = class_correct / class_total
            out = [str(item) for item in (epoch, train_loss, valid_loss)]
            out += [str(item) for item in stats]
            fh.write(",".join(out))
            fh.write("\n")
            fh.flush()

            msg = f'{split},{batchsize},{ceta}:{epoch}: Training Loss: {train_loss:.6f}: Validation Loss: {valid_loss:.6f} Individual stats: '
            msg += " ".join([f'{item:.3f}' for item in stats])
            print(msg)

            # save model to file.
            # This file can be later copied back locally if you are running remotely
            # for analysis.
            if valid_loss <= valid_loss_min:
                # save parameters for later loading
                torch.save(classifier.state_dict(), f'{cfilename}_lowest.pt')
                valid_loss_min = valid_loss
            torch.save(classifier.state_dict(), f'{cfilename}.pt')

        classifier.load_state_dict(torch.load(f'{cfilename}_lowest.pt'))
        class_correct, class_total, test_loss = classifier.test(
            test_loader, closs, ctag)
        fh.write('#############\n')
        fh.write(f'class_correct={class_correct},class_total={class_total},test_loss={test_loss}')


if __name__ == '__main__':
    import sys
    splits = [int(item) for item in sys.argv[1].split(",")]
    parallel = 4
    gridsearch_args = None
    if len(sys.argv) > 2:
        ceta, batchsize = sys.argv[2].split(",")
        gridsearch_args = [(float(ceta), int(batchsize))]

        if len(sys.argv) > 3:
            parallel = int(sys.argv[3])

    if parallel:
        print(f"Running parallel with {parallel}")
        multiprocessing.set_start_method('spawn')

    args = []
    if not gridsearch_args:
        gridsearch_args = [
            (0.001, 500),
            (0.0001, 500),
            (0.00001, 500),
            (0.001, 100),
            (0.0001, 100),
            (0.00001, 100),
            (0.001, 50),
            (0.0001, 50),
            (0.00001, 50),
        ]

    for ceta, batchsize in gridsearch_args:
        for split in splits:
            print(f"Preparing {split}")
            splitargs = (split, ceta, batchsize)
            args.append(splitargs)

    if parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
            for _ in executor.map(do_run, args):
                pass
    else:
        for split in args:
            do_run(split)
