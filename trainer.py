from visual import Logger

import os
import os.path as osp
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F

from data import CUB200_loader
import cv2

from models import DeepCooc, Resnet152, Resnet50

import argparse

parser = argparse.ArgumentParser(description = 'Training arguments')
parser.add_argument('--cuda', default = True, type = bool, help = 'use cuda to train')
parser.add_argument('--lr', default = 0.001, type = float, help = 'initial learning rate')
parser.add_argument('--lr_decay_steps', default = None, nargs = '+', type = int, help = 'lr decays for each steps')
parser.add_argument('--batch_size', default = 32, type = int, help = '# of sample to learn at one iteration')
parser.add_argument('--start_iter', default = 0, type = int, help = 'start iteration')
parser.add_argument('--end_iter', default = 50000, type = int, help = 'end iteration')
parser.add_argument('--model_select', default = 'Deepcooc', type = str, help = ' --model_select: ["deepcooc", "resnet152"]')
parser.add_argument('--exp_name', default = 'Deepcooc_CUB_resnet152_3', type = str, help = 'name of experiment')
args = parser.parse_args()
args.lr_decay_steps = [30, 60]
decay_steps = None

net = None
if args.model_select == 'Deepcooc':
    print(" [*] Deepcooc network is selected")
    net = DeepCooc(num_classes = 200)
elif args.model_select == 'resnet152':
    print(" [*] Resnet152 is selected")
    net = Resnet152(num_classes = 200)
elif args.model_select == 'resnet50':
    print(" [*] Resnet50 is selected")
    net = Resnet50(num_classes = 200)
else:
    raise ValueError(" [!] Model does not selected properly")

if args.cuda and torch.cuda.is_available():
    print(" [*] Set cuda: True")
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
else:
    print(" [*] Set cuda: False")

logger = Logger('./visual/' + args.exp_name)
opt = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), 
                lr = args.lr, momentum = 0.9)
#opt = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), 
#                lr = args.lr, weight_decay = 0.00005)
criterion = nn.CrossEntropyLoss()

def train():
    net.train()

    print(" [*] Loading dataset...")
    batch_iterator = None

    trainset = CUB200_loader(os.getcwd() + '/data/CUB_200_2011', split = 'train')
    trainloader = data.DataLoader(trainset, batch_size = args.batch_size,
                    shuffle = True, collate_fn = trainset.CUB_collate, num_workers = 4)
    testset = CUB200_loader(os.getcwd() + '/data/CUB_200_2011', split = 'test')
    testloader = data.DataLoader(testset, batch_size = args.batch_size,
                    shuffle = False, collate_fn = testset.CUB_collate, num_workers = 4)

    print(" [*] Loading done, training start")
    epoch = -1
    loss = 0
    old_loss = 999.
    old_acc = 0.
    steps = 0 # need to decay lr
    epoch_size = len(trainset) // args.batch_size

    start_iter = args.start_iter
    global iteration
    for _iter in range(start_iter, args.end_iter + 1):
        iteration = _iter
        if (not batch_iterator) or (iteration % epoch_size == 0):
            batch_iterator = iter(trainloader)

        if iteration % epoch_size == 0:
            epoch += 1

        if epoch in args.lr_decay_steps:
            steps += 1
            adjust_learning_rate(opt, 0.1, steps, args.lr)
    
        images, labels = next(batch_iterator)
        if args.cuda:
            images = images.cuda()
            labels = labels.cuda()

        t0 = time.time()

        logits, l1_loss = net(images)
        opt.zero_grad()

        c_loss = criterion(logits, labels)
        loss = l1_loss + c_loss
        loss.backward()
        
        opt.step()
        t1 = time.time()

        logger.scalar_summary('train_c_loss', c_loss.item(), iteration + 1)
        logger.scalar_summary('train_l1_loss', l1_loss.item(), iteration + 1)

        if (iteration % 20) == 0:
            print(" [*] Epoch[%d], Iter %d || c_Loss: %.4f || l1_loss: %.4f || gamma: %.4f || Timer: %.4fsec"%(epoch, iteration, c_loss.item(), l1_loss.item(), net.module.gamma.item(), (t1 - t0)))

        if (iteration % 1000) == 0:
            net.eval()

            test_losses = []
            correct = 0.
            with torch.no_grad():
                for test_images, test_labels in testloader:
                    
                    if args.cuda:
                        test_images = test_images.cuda()
                        test_labels = test_labels.cuda()

                    test_logits, _ = net(test_images)
                    test_loss = criterion(test_logits, test_labels)

                    test_losses.append(test_loss.item())
                    test_preds = test_logits.data.max(1, keepdim = True)[1]
                    correct += test_preds.eq(test_labels.data.view_as(test_preds)).long().cpu().sum()

            test_loss = np.mean(test_losses)
            test_acc = 100. * float(correct) / len(testset)

            print("  [*] Test loss: %.4f, Test acc: %.4f"%(test_loss, test_acc))
            logger.scalar_summary('Test loss', test_loss, iteration + 1)
            logger.scalar_summary('Test acc', test_acc, iteration + 1)

            if test_loss < old_loss or test_acc > old_acc or (iteration % 10000) == 0:
                print("  [*] Save ckpt, iter: %d at ckpt/"%iteration)
                file_path = 'ckpt/%s_%d_%s.pth'%(args.exp_name, iteration, str(args.lr))

                torch.save(net.state_dict(), file_path)
                if test_loss < old_loss:
                    old_loss = test_loss

                if test_acc > old_acc:
                    old_add = test_acc

            net.train()
    print(" [*] Training end")

def adjust_learning_rate(optimizer, gamma, steps, _lr):
    lr = _lr * (gamma ** (steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    train()
