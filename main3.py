import torch
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


######################################################################################
#    ESEGUO MLP CON:
#    - DATASET NORMALIZZATO E 12 Labels
#    - 25 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 32] 12
#    - optimizer SGD

class CustomDataset(Dataset):
    def __init__(self, Features, Labels,  transform=None, target_transform=None):
        self.labels = Labels
        self.features= Features
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

input_size = 531
hidden_sizes = [256, 128, 32]
output_size = 12

class NetMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.fl1 = nn.Linear(input_size, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fl4 = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = F.relu(self.fl1(x))
        x = F.relu(self.fl2(x))
        x = F.relu(self.fl3(x))
        x = self.fl4(x)
        return x

#classes = ('FB', 'FL', 'TW')

f = h5py.File('12LabelsNormalized.h5', 'r')

"""

Features = f['train/features']
Labels= f['train/labels']

# trasform = none perché escono già come Tensori

trainingSet = CustomDataset(Features,Labels)
trainDataloader = DataLoader(trainingSet, batch_size=117, shuffle=True)

net = NetMLP(input_size, hidden_sizes, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(25):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainDataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 18 == 17:    # print every 39 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 18))
            running_loss = 0.0

print('Finished Training')

PATH = './last.pth'
torch.save(net.state_dict(), PATH)

"""

validSet = f['valid']
Features = validSet['features']
Labels= validSet['labels']

validationSet = CustomDataset(Features,Labels)
validDataloader = torch.utils.data.DataLoader(validationSet, batch_size=60, shuffle=False)

# Salvataggio
net = NetMLP(input_size, hidden_sizes, output_size)
PATH = './last.pth'
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in validDataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 7020 validation images: %d %%' % (100 * correct / total))

"""

testSet = f['test']
Features = testSet['features']
Labels= testSet['labels']

testSet = CustomDataset(Features,Labels)
testDataloader = torch.utils.data.DataLoader(validationSet, batch_size=60, shuffle=False)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in validDataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 7020 test images: %d %%' % (100 * correct / total))

"""
