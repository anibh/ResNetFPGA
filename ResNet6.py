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
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.linear = nn.Linear(128*block.expansion, num_classes)

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

##########################  ResNet18 Model ##########################################
ResNet14 = ResNet(Block, [2,2,2])
print(ResNet14)
print("Total number of parameters: ", ResNet14.num_params())

###################################
# Training

import torch.optim as optim

# Loss function: Cross Entropy
criterion = nn.CrossEntropyLoss()

# Configuring stochastic gradient descent optimizer
optimizer = optim.SGD(ResNet14.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-4)

# Each epoch will go over training set once; run two epochs
for epoch in range(epochs):

    running_loss = 0.0

    # iterate over the training set
    for i, data in enumerate(trainloader, 0):
        # Get the inputs
        inputs, labels = data

        # Clear the parameter gradients
        optimizer.zero_grad()

        #################################
        # forward + backward + optimize

        # 1. evaluate the current network on a minibatch of the training set
        outputs = ResNet14(inputs)
        # print('Epoch: ', epoch)
        # print('Enumeration: ', i)
        #print(outputs)

        # 2. compute the loss function
        loss = criterion(outputs, labels)

        # 3. compute the gradients
        loss.backward()

        # 4. update the parameters based on gradients
        optimizer.step()

        # Update the average loss
        running_loss += loss.item()

        # Print the average loss every 256 minibatches ( == 16384 images)
        if i % 256 == 255:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 256))
            running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():  # this tells PyTorch that we don't need to keep track
        # of the gradients because we aren't training
        for data in testloader:
            images, labels = data
            outputs = ResNet14(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Epoch %d: Accuracy of the network on the %d test images: %d/%d = %f %%' % (
        epoch + 1, total, correct, total, (100 * correct / total)))

print('Finished Training!')

###################################
# Let's look at some test images and see what our trained network predicts for them

if (showImages > 0):
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images = images[0:4]
    labels = labels[0:4]
    outputs = ResNet14(images)
    _, predicted = torch.max(outputs.data, 1)

    print('Predicted: ', ' '.join('%10s' % classes[predicted[j]] for j in range(4)))

    imshow(torchvision.utils.make_grid(images))

##################################
# Let's comptue the total accuracy across the training set

correct = 0
total = 0
with torch.no_grad():  # this tells PyTorch that we don't need to keep track
    # of the gradients because we aren't training
    for data in trainloader:
        images, labels = data
        outputs = ResNet14(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d training images: %f %%' % (total, (100 * correct / total)))

##################################
# Now we want to compute the total accuracy across the test set

correct = 0
total = 0
with torch.no_grad():  # this tells PyTorch that we don't need to keep track
    # of the gradients because we aren't training
    for data in testloader:
        images, labels = data
        outputs = ResNet14(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test images: %d/%d = %f %%' % (total, correct, total, (100 * correct / total)))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = ResNet14(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %10s : %f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

# Save the ResNet18 trained model
torch.save(ResNet14, 'ResNet14_1.pth')
print("Saved ResNet14 trained model in ResNet14_1.pth file in home directory")

################################ ResNet18 Model End #################################################################


