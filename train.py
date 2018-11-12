import numpy as np
import matplotlib.pyplot as plt
import os, random
import json
import argparse
import torch
from torch import nn
import seaborn as sns
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict

import time

def do_deep_learning(args, epochs, criterion, cuda=True):
    dataloaders, image_datasets = initialize_data(args.data_dir)
    model = build_network(args.arch)
    optimizer = optim.SGD(model.classifier.parameters(), lr=float(args.learning_rate))
    if cuda:
        model.cuda()
    else:
        model.cpu()

    running_loss = 0
    accuracy = 0

    start = time.time()
    print('Training started')

    for e in range(epochs):

        for mode in ['training', 'validation']:   
            if mode == 'training':
                model.train()
            else:
                model.eval()

            pass_count = 0

            for data in dataloaders[mode]:
                pass_count += 1
                inputs, labels = data
                if cuda == True:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                # Forward
                output = model.forward(inputs)
                loss = criterion(output, labels)
                # Backward
                if mode == 'training':
                    loss.backward()
                    optimizer.step()                

                running_loss += loss.item()
                ps = torch.exp(output).data
                equality = (labels.data == ps.max(1)[1])
                accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()

            if mode == 'training':
                print("\nEpoch: {}/{} ".format(e+1, epochs),
                      "\nTraining Loss: {:.4f}  ".format(running_loss/pass_count))
            else:
                print("Validation Loss: {:.4f}  ".format(running_loss/pass_count),
                  "Accuracy: {:.4f}".format(accuracy))

            running_loss = 0

    time_elapsed = time.time() - start
    print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))
    save_checkpoint(model, args, image_datasets)
    
def build_network(arch):
    model = getattr(models, arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    feature_num = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(feature_num, 1024)),
                              ('drop', nn.Dropout(p=0.5)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(1024, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    return model

def initialize_data(directory):
    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomRotation(30),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    validataion_transforms = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                                      [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])]) 

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'training': datasets.ImageFolder(directory + '/train', transform=training_transforms),
        'validation': datasets.ImageFolder(directory + '/valid', transform=validataion_transforms),
        'testing': datasets.ImageFolder(directory + '/test', transform=testing_transforms)
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'validation': torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64)
    }
    return dataloaders, image_datasets

def save_checkpoint(model, args, image_datasets):
    model.class_to_idx = image_datasets['training'].class_to_idx

    checkpoint = {
        'arch': args.arch,
        'classifier' : model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(checkpoint, 'checkpoint.pth')

def arg_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='vgg19')
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.01')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='10')
    parser.add_argument('--gpu', action="store_true", default=True)
    return parser.parse_args()

args = arg_parser()
criterion = nn.CrossEntropyLoss()
epochs = int(args.epochs)
cuda = True if torch.cuda.is_available() and args.gpu else False
do_deep_learning(args, epochs, criterion, cuda)