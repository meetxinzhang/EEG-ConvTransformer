# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/8 21:30
 @name: 
 @desc:
"""

import numpy as np
import scipy.io as sio
import torch
import os

import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader,random_split


def test(net, Testloader, criterion, is_cuda=True):
    running_loss = 0.0
    evaluation = []
    for i, data in enumerate(Testloader, 0):
        input_img, labels = data
        input_img = input_img.to(torch.float32)
        if is_cuda:
            input_img = input_img.cuda()
        outputs = net(input_img)
        _, predicted = torch.max(outputs.cpu().data, 1)
        evaluation.append((predicted == labels).tolist())
        loss = criterion(outputs, labels.cuda())
        running_loss += loss.item()
    running_loss = running_loss / (i + 1)
    evaluation = [item for sublist in evaluation for item in sublist]
    running_acc = sum(evaluation) / len(evaluation)
    return running_loss, running_acc


def train_validate(model, trainloader, testloader, n_epoch=30, opti='SGD', learning_rate=0.0001, is_cuda=True,
                   print_epoch=5, verbose=False):
    if is_cuda:
        net = model().cuda()
    else:
        net = model()

    criterion = nn.CrossEntropyLoss()

    if opti == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    elif opti == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    else:
        print("Optimizer: " + optim + " not implemented.")

    for epoch in range(n_epoch):
        running_loss = 0.0
        evaluation = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(torch.float32).cuda())
            _, predicted = torch.max(outputs.cpu().data, 1)
            evaluation.append((predicted == labels).tolist())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss = running_loss / (i + 1)
        evaluation = [item for sublist in evaluation for item in sublist]
        running_acc = sum(evaluation) / len(evaluation)
        validation_loss, validation_acc = Test_Model(net, testloader, criterion, True)

        if epoch % print_epoch == (print_epoch - 1):
            print('[%d, %3d]\tloss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
                  (epoch + 1, n_epoch, running_loss, running_acc, validation_loss, validation_acc))
    if verbose:
        print('Finished Training \n loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
              (running_loss, running_acc, validation_loss, validation_acc))

    return (running_loss, running_acc, validation_loss, validation_acc)