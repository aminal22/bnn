from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import nin
from torch.autograd import Variable
import util

# Save the model state
def save_state(model, best_acc):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = state['state_dict'].pop(key)
    torch.save(state, 'models/nin.pth.tar')

# Training process
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        # Binarization process for weights
        bin_op.binarization()
        
        # Forward pass
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        
        # Backward pass
        loss = criterion(output, target)
        loss.backward()
        
        # Restore weights and update gradients
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return

# Testing process
def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    bin_op.restore()
    acc = 100. * float(correct) / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return

# Adjust learning rate at specific epochs
def adjust_learning_rate(optimizer, epoch):
    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

if __name__=='__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data/', help='dataset path')
    parser.add_argument('--arch', action='store', default='nin', help='network architecture: nin')
    parser.add_argument('--lr', action='store', default='0.01', help='initial learning rate')
    parser.add_argument('--pretrained', action='store', default=None, help='path to pretrained model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
    args = parser.parse_args()
    print('==> Options:', args)

    # Set the random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Prepare dataset with torchvision
    print('==> Preparing dataset ...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Define classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Build model
    print('==> Building model', args.arch, '...')
    if args.arch == 'nin':
        model = nin.Net()
    else:
        raise Exception(args.arch + ' is currently not supported')

    # Load pretrained model if available
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()
    else:
        print('==> Loading pretrained model from', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)

    # Define optimizer and loss function
    base_lr = float(args.lr)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()

    # Define binarization operator
    bin_op = util.BinOp(model)

    # Evaluation mode if specified
    if args.evaluate:
        test()
        exit(0)

    # Start training
    for epoch in range(1, 320):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()
