import torch
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, Features, Labels1, Labels2,  transform=None, target_transform=None):
        self.labels1 = Labels1
        self.labels2 = Labels2
        self.features= Features
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels1)

    def __getitem__(self, idx):
        image = self.features[idx]
        labels1 = self.labels1[idx]
        labels2 = self.labels2[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels1 = self.target_transform(labels1)
            labels2 = self.target_transform(labels2)
        return image, labels1, labels2

input_size = 531
hidden_sizes = [256, 128, 32]
output_size = 4

class NetMLPLatent(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.latent_size = hidden_sizes[2]
        self.fl1 = nn.Linear(input_size + self.latent_size, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fl4 = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x, h):
        x = torch.cat(x, h, 0)   # Concatenates the h tensor to input x.
        x = F.relu(self.fl1(x))  # Note above how the size of fl1 had to be changed.
        x = F.relu(self.fl2(x))
        latent = F.relu(self.fl3(x))
        x = self.fl4(latent)
        return x, latent

class NetMLPUnrolled(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.share1 = NetMLPLatent(input_size, hidden_sizes, output_size)
        self.share2 = NetMLPLatent(input_size, hidden_sizes, output_size)

    def forward(self, x):
        share1, latent = self.share1(x, torch.zeros([self.latent_size]))
        share2, _ = self.share2(x, latent)
        return share1, share2

f = h5py.File('12LabelsNormalized.h5', 'r')
f1 = h5py.File('doubleLabels.h5', 'r')

Features = f['train/features']
Labels1 = f1['train/labels/share1']
Labels2 = f1['train/labels/share2']

# trasform = none perché escono già come Tensori

trainingSet = CustomDataset(Features, Labels1, Labels2)
trainDataloader = DataLoader(trainingSet, batch_size=117, shuffle=True)

net = NetMLPUnrolled(input_size, hidden_sizes, output_size)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters())
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)

for epoch in range(45):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainDataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels1, labels2 = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output1, output2 = net(inputs)
        loss1 = criterion(output1, labels1)
        loss2 = criterion(output2, labels2)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 18 == 17 :
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 18))
            running_loss = 0.0

print('Finished Training')

PATH = './last.pth'
torch.save(net.state_dict(), PATH)

"""

# Salvataggio
net = NetMLP(input_size, hidden_sizes, output_size1, output_size2)
PATH = './Adam2HideDoubleSum78.pth'
net.load_state_dict(torch.load(PATH))

Features = f['valid/features']
Labels1 = f1['valid/labels/share1']
Labels2 = f1['valid/labels/share2']

validationSet = CustomDataset(Features, Labels1, Labels2)
validDataloader = torch.utils.data.DataLoader(validationSet, batch_size=60, shuffle=False)

correct = 0
total = 0
a = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in validDataloader:
        images, labels1, labels2 = data
        # calculate outputs by running images through the network
        output1, output2 = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted1 = torch.max(output1.data, 1)
        _, predicted2 = torch.max(output2.data, 1)
        total += labels1.size(0)
        for i in range(len(predicted1)):
            if predicted1[i] == labels1[i] and predicted2[i] == labels2[i]:
                correct += 1

print('Accuracy of the network on the 7020 validation images: %d %%' % (100 * correct / total))

Features = f['test/features']
Labels1 = f1['test/labels/share1']
Labels2 = f1['test/labels/share2']

testSet = CustomDataset(Features, Labels1, Labels2)
testDataloader = torch.utils.data.DataLoader(testSet, batch_size=60, shuffle=False)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testDataloader:
        images, labels1, labels2 = data
        # calculate outputs by running images through the network
        output1, output2 = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted1 = torch.max(output1.data, 1)
        _, predicted2 = torch.max(output2.data, 1)
        total += labels1.size(0)
        for i in range(len(predicted1)):
            if predicted1[i] == labels1[i] and predicted2[i] == labels2[i]:
                correct += 1

print('Accuracy of the network on the 7020 test images: %d %%' % (100 * correct / total))

"""


