import torch
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

######################################################################################
#    ESEGUO MLP CON:
#    - DATASET NORMALIZZATO E 3 Labels
#    - 20 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 32] 3
#    - optimizer Adam ---> 100%
#    100Adam.pth

#    ESEGUO MLP CON:
#    - DATASET NORMALIZZATO E 3 Labels
#    - 20 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 32] 3
#    - optimizer SGD ---> 100%
#    100SGD.pth

#    ESEGUO MLP CON:
#    - DATASET NORMALIZZATO CON 2 FEATURES E 3 Labels
#    - 20 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 521 [256, 128, 32] 3
#    - optimizer Adam ---> 92.96% sul testset
#    92.96_D3.pth

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

input_size = 521 #531
hidden_sizes = [256, 128, 32]
output_size = 3

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

f = h5py.File('3LabelsNormalized.h5', 'r')
f1 = h5py.File('2FeaturesNormalized.h5','r')

Features = f1['train/features']
Labels= f['train/labels']

# trasform = none perché escono già come Tensori

trainingSet = CustomDataset(Features,Labels)
trainDataloader = DataLoader(trainingSet, batch_size=117, shuffle=True)

trainingSet1 = CustomDataset(Features,Labels)
trainDataloader1 = DataLoader(trainingSet, batch_size=117, shuffle=False)

Features1 = f1['valid/features']
Labels1= f['valid/labels']

validationSet = CustomDataset(Features1,Labels1)
validDataloader = torch.utils.data.DataLoader(validationSet, batch_size=60, shuffle=False)

#dataiter = iter(trainDataloader)
#images, labels = dataiter.next()

net = NetMLP(input_size, hidden_sizes, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter("runs")
max = 0

for epoch in range(20):  # loop over the dataset multiple times

    print('Running Epoch: ', epoch)
    
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

    running_loss_train = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in trainDataloader1:
            inputs, labels = data

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss_train += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    Accuracy_Train = 100 * correct / total
    running_loss_train = running_loss_train/ 180

    running_loss_valid = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in validDataloader:
            inputs, labels = data

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss_valid += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    Accuracy_Valid = 100 * correct / total

    if Accuracy_Valid > max:
        max = Accuracy_Valid
        PATH = './last.pth'
        torch.save(net.state_dict(), PATH)

    running_loss_valid = running_loss_valid / 120

    writer.add_scalars('Loss', {'trainset': running_loss_train,'validset': running_loss_valid}, epoch+1)
    writer.add_scalars('Accuracy', {'trainset': Accuracy_Train,'validset': Accuracy_Valid}, epoch+1)

writer.close()
print("Max Accuracy in validtest: ", max)
print('Finished Training')

"""
# Salvataggio
net = NetMLP(input_size, hidden_sizes, output_size)
PATH = './last.pth'
net.load_state_dict(torch.load(PATH))

"""

Features2 = f1['test/features']
Labels2= f['test/labels']

testSet = CustomDataset(Features2,Labels2)
testDataloader = torch.utils.data.DataLoader(testSet, batch_size=60, shuffle=False)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testDataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 7020 test images: %.2f %%' % (100 * correct / total))



