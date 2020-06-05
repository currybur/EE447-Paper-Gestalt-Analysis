# -*- coding:utf-8 -*-

import os
import shutil

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import Dataset

from torchvision import models
from torchvision import transforms

from sklearn.metrics import classification_report

from predict import predict_page, get_nn_model

# INPUT_DIR = '../input/'
INPUT_DIR = '../dataset/'
OUTPUT_DIR = '../output/'
BATCH_SIZE = 32


class MyDataset(Dataset):
    def __init__(self, is_train=True, transform=None):
        folder = INPUT_DIR + ('train/' if is_train else 'test/')
        # model = get_nn_model('overall')
        path = []
        label = []
        l1 = len(os.listdir(folder + 'conference/'))
        l2 = len(os.listdir(folder + 'workshop/'))
        data_size = min(l1,l2)

        for conf, lab in zip(['conference/', 'workshop/'], [1, 0]):
            # print(conf, lab)
            ct = 0
            for p in os.listdir(folder + conf):
                # if ct >= data_size:break  # data pruning
                ct += 1
                path.append(folder + conf + p)
                label.append(lab)
                if conf == 'workshop/':  # data augmentation
                    for i in range(int(l1/l2)-1):
                        path.append(folder + conf + p)
                        label.append(lab)
                        # label.append(predict_page(folder + conf + p, model)/100.0)
        self.path = path
        self.label = label
        self.transform = transform
        # print(len(path), len(label))

    def __getitem__(self, index):
        img_path, label = self.path[index], self.label[index]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.path)


def load_data(batch_size):
    train_loader = torch.utils.data.DataLoader(
        MyDataset(
            is_train=True,
            transform=transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        MyDataset(
            is_train=False,
            transform=transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])),
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(trainloader, model, criterion, optimizer):
    model.train()

    all_cnt = 0.0
    true_cnt = 0.0
    loss_sum = 0.0
    l_pred = []
    l_true = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # targets = torch.tensor(targets, dtype=torch.float32)
        inputs, targets = inputs.float().cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = outputs.detach().cpu().numpy()
        # pred = probs
        pred = np.argmax(probs, axis=1)
        # print(probs)
        # pred = np.array([1 if i > 0 else 0 for i in probs])
        # pred.reshape((-1,1))

        ytrue = targets.cpu().numpy()
        for i in pred:
            l_pred.append(i)
        for i in ytrue:
            l_true.append(i)


        all_cnt += targets.shape[0]
        true_cnt += (ytrue == pred).sum()
        loss_sum += float(loss)
    l_pred = np.array(l_pred)
    l_true = np.array(l_true)
    return loss_sum, true_cnt / all_cnt, l_pred.ravel(), l_true.ravel()


def test(testloader, model, criterion):
    model.eval()

    all_cnt = 0.0
    true_cnt = 0.0
    loss_sum = 0.0
    l_pred = []
    l_true = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # targets = torch.tensor(targets, dtype=torch.float32)
        inputs, targets = inputs.float().cuda(), targets.cuda(non_blocking=True)
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        probs = outputs.detach().cpu().numpy()
        # pred = probs

        pred = np.argmax(probs, axis=1)
        # pred = np.array([1 if i > 0 else 0 for i in probs])
        # pred.reshape((-1,1))
        ytrue = targets.cpu().numpy()
        for i in pred:
            l_pred.append(i)
        for i in ytrue: 
            l_true.append(i)

        all_cnt += targets.shape[0]
        true_cnt += (ytrue == pred).sum()
        loss_sum += float(loss)
    l_pred = np.array(l_pred)
    l_true = np.array(l_true)
    return loss_sum, true_cnt / all_cnt, l_pred.ravel(), l_true.ravel()


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='ckpt_res18_ov3_bl.pth.tar',
                    bestname='bmdl_res18_ov3_bl.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, bestname))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [1, 2, 3, 4, 5]:
        state['lr'] *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    print('execute nn_train.py ...')
    ckp_dir = OUTPUT_DIR + 'nn_output/'
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)

    state = {'lr': 0.001}

    # model = models.vgg19_bn(pretrained=False)
    # num_ftrs = model.classifier[-1].in_features
    # model.classifier[-1] = nn.Linear(num_ftrs, 2)

    # model = models.densenet121(pretrained=False)
    # num_ftrs = model.classifier.in_features
    # model.classifier = nn.Linear(num_ftrs, 2)

    # model = models.resnext50_32x4d(pretrained=False)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)

    model = models.resnet18(pretrained=False)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.fc = nn.Linear(model.fc.in_features, 2)

    # model = models.inception_v3(pretrained=False)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)

    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    train_loader, test_loader = load_data(BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=state['lr'], momentum=0.9, weight_decay=1e-4)

    best_acc = 0  # best test accuracy
    for epoch in range(10):
        print('-'*50, 'epoch ', epoch)
        adjust_learning_rate(optimizer, epoch)
        train_loss, train_acc, train_pred, train_true = train(train_loader, model, criterion, optimizer)
        test_loss, test_acc, test_pred, test_true = test(test_loader, model, criterion)
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        print(round(train_loss, 5), round(train_acc, 5), round(test_loss, 5), round(test_acc, 5), round(best_acc, 5))
        print(classification_report(train_true, train_pred))
        print(classification_report(test_true, test_pred))

        if not is_best:
            print('early stop.')
            break
            # continue
        else:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'lr': state['lr']
            }, is_best, checkpoint=ckp_dir)




