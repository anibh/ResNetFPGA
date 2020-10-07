# ESE 587, Stony Brook University
# Handout Code for PyTorch Warmup 1

# Based on PyTorch tutorial code from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

############################
# Parameters you can adjust
showImages = 0  # Will show images as demonstration if = 1
batchSize = 64  # The batch size used for learning
learning_rate = 0.001  # Learning rate used in SGD
momentum = 0.9  # Momentum used in
epochs = 30  # Number of epochs to train for

############################################
# Set up our training and test data
# The torchvision package gives us APIs to get data from existing datasets like MNST
# The "DataLoader" function will take care of downloading the test and training data

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#######################################
# Let's look at a few random images from the training data

import matplotlib.pyplot as plt
import numpy as np


# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # undo normalization
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if (showImages > 0):
    # Grab random images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    images = images[0:4]
    labels = labels[0:4]

    # print labels
    print(' '.join('%s' % classes[labels[j]] for j in range(4)))
    # Show images
    imshow(torchvision.utils.make_grid(images))

##################################
# Define our network

import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    # Some simple code to calculate the number of parametesr
    def num_params(self):
        numParams = 0
        for param in ResNet14.parameters():
            thisLayerParams = 1
            for s in list(param.size()):
                thisLayerParams *= s
            numParams += thisLayerParams

        return numParams


ResNet14 = torch.load('ResNet14_0.pth')
ResNet14.eval()
print(ResNet14)
print("Total number of parameters: ", ResNet14.num_params())

fout = open("weights.h", "a")
for name, param in ResNet14.named_parameters():
    if "conv" in name:
        name = name.replace(".", "_")
        string = 'float '+name
        for num in list(param.size()):
            string += '['+str(num)+']'
        string += ' = { '
        n = 0
        for ch in param:
            string += '{'
            l = 0
            for kernel in ch:
                string += '{'
                m = 0
                for row in kernel:
                    string += '{'
                    k = 0
                    for num in row:
                        string += str(num.item())
                        if(k != row.size(0) - 1): string += ', '
                        k += 1
                    string += '}'
                    if(m != kernel.size(0) - 1): string += ', '
                    m += 1
                string += '}'
                if(l != ch.size(0) - 1): string += ', '
                l += 1
            string += '}'
            if(n != param.size(0) - 1): string += ', '
            n += 1
        string += '};\n'
        #print(string)
    elif "bn" in name:
        name = name.replace(".", "_")
        string = 'float '+name
        for num in list(param.size()):
            string += '['+str(num)+']'
        string += ' = {'
        k = 0
        for num in param:
            string += str(num.item())
            if(k != param.size(0) - 1): string += ', '
            k += 1
        string += '};\n'
        #print(string)
    elif "shortcut" in name:
        if "shortcut.0" in name:
            name = name.replace(".", "_")
            string = 'float '+name
            for num in list(param.size()):
                string += '['+str(num)+']'
            string += ' = { '
            for ch in param:
                string += '{'
                l = 0
                for kernel in ch:
                    string += '{'
                    m = 0
                    for row in kernel:
                        string += '{'
                        k = 0
                        for num in row:
                            string += str(num.item())
                            if(k != row.size(0) - 1): string += ', '
                            k += 1
                        string += '}'
                        if(n != kernel.size(0) - 1): string += ', '
                    string += '}'
                    if(m != ch.size(0) - 1): string += ', '
                    m += 1
                string += '}'
                if(l != param.size(0) - 1): string += ', '
                l += 1
            string += '};\n'
            #print(string)
        else:
            name = name.replace(".", "_")
            string = 'float '+name
            for num in list(param.size()):
                string += '['+str(num)+']'
            string += ' = {'
            k = 0
            for num in param:
                string += str(num.item())
                if(k != param.size(0) - 1): string += ', '
                k += 1
            string += '};\n'
    elif "linear" in name:
        if "weight" in name:
            name = name.replace(".", "_")
            string = 'float '+name
            for num in list(param.size()):
                string += '['+str(num)+']'
            string += ' = {'
            m = 0
            for row in param:
                string += '{'
                k = 0
                for num in row:
                    string += str(num.item())
                    if(k != row.size(0) - 1): string += ', '
                    k += 1
                string += '}'
                if(m != param.size(0) - 1): string += ', '
                m += 1
            string += '};\n'
        else:
            name = name.replace(".", "_")
            string = 'float '+name
            for num in list(param.size()):
                string += '['+str(num)+']'
            string += ' = {'
            k = 0
            for num in param:
                string += str(num.item())
                if(k != param.size(0) - 1): string += ', '
                k += 1
            string += '};\n'
    fout.write(string)
    print(name)
fout.close()

n = 0
fout = open("inputImages.h", "a")
stringLabel = 'int label[64] = {'
stringBatch = 'float batch[64][3][32][32] = {'
for imageBatch, labelBatch in testloader:
    if(n > 0): break
    a = 0
    for label in labelBatch:
        if(a == 2): break
        stringLabel += str(label.item())
        if(a != labelBatch.size(0) - 1): stringLabel += ', '
        a += 1
    print('label'+str(n))
    print('batch'+str(n))
    j = 0
    for image in imageBatch:
        if(j == 2): break
        stringBatch += '{'
        l = 0
        for plane in image:
            stringBatch += '{'
            m = 0
            for row in plane:
                stringBatch += '{ '
                k = 0
                for x in row:
                    stringBatch += str(x.item())
                    if(k != row.size(0) - 1): stringBatch += ', '
                    k += 1
                stringBatch += '}'
                if(m != plane.size(0) - 1): stringBatch += ', '
                m += 1
            stringBatch += '}'
            if(l != image.size(0) -1): stringBatch += ', '
            l += 1
        stringBatch += '}'
        if(j != imageBatch.size(0) - 1): stringBatch += ', '
        j += 1
    n += 1
stringLabel += '};\n'
stringBatch += '};\n'
fout.write(stringLabel)
fout.write(stringBatch)

