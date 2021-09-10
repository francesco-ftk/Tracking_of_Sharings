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
### ESEGUO METODO IN "2share_Direc_12Labels.py" PER FARE PLOT DI ACCURACY E LOSS
### SUL TRAINSET E VALIDSET PER 150 EPOCHE DI ADDESTRAMENTO.
### DAL PLOT RISULTA CHE L'OVERFIT INIZA FRA LA 20-ESIMA E 40-ESIMA EPOCA
### DI ADDESTRAMENTO. L'ACCURATEZZA DEL VALIDSET AUMENTA FINO A STABILIZZARSI
### VERSO LA 60-EPOCA.

#    ESEGUO MLP CON:
#    - DATASET NORMALIZZATO E 12 Labels
#    - 60 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 64] 12
#    - optimizer SGD con Nesterov Momentum ---> 75% con loss di 0.562
#    75Nesterov2.pth

#    ESEGUO MLP CON:
#    - DATASET NORMALIZZATO E 12 Labels
#    - 45 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 64] 12
#    - optimizer Adam ---> 79% con loss di 0.392
#    79Adam2.pth

### RETE CON LA MIGLIORE ACCURATEZZA SUL VALIDSET:
#    ESEGUO METODO DIRECT12:
#    - DATASET NORMALIZZATO E 12 LABELS
#    - 77/80 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 32] 3/4
#    - Adam ---> 80.09% sul valid, 81.15% sul test
#    81.15Adam2.pth


batch_size_train = 117
batch_size_valid_and_test = 60

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
hidden_sizes = [256, 128, 64]
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

f = h5py.File('12LabelsNormalized.h5', 'r')

"""
Features_test = f['train/features']
Labels_test = f['train/labels']

# trasform = none perché escono già come Tensori

trainingSet = CustomDataset(Features_test, Labels_test)
trainDataloader = DataLoader(trainingSet, batch_size=batch_size_train, shuffle=True)

trainingSet1 = CustomDataset(Features_test, Labels_test)
trainDataloader1 = DataLoader(trainingSet1, batch_size=batch_size_train, shuffle=False)

Features = f['valid/features']
Labels1 = f['valid/labels']

validationSet = CustomDataset(Features, Labels1)
validDataloader = torch.utils.data.DataLoader(validationSet, batch_size=batch_size_valid_and_test, shuffle=False)

net = NetMLP(input_size, hidden_sizes, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# Writer will output to ./runs/ directory by default
writer = SummaryWriter("runs")
max = 0

for epoch in range(80):

    print('Running Epoch: ', epoch)

    for i, data in enumerate(trainDataloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

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
print('Finished')

"""

# Salvataggio
net = NetMLP(input_size, hidden_sizes, output_size)
PATH = './81.15Adam2.pth'
net.load_state_dict(torch.load(PATH))

validSet = f['valid']
Features = validSet['features']
Labels= validSet['labels']

validationSet = CustomDataset(Features,Labels)
validDataloader = torch.utils.data.DataLoader(validationSet, batch_size=60, shuffle=False)

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

print('Accuracy of the network on the 7020 validation images: %.2f %%' % (100 * correct / total))


testSet = f['test']
Features = testSet['features']
Labels= testSet['labels']

testSet = CustomDataset(Features,Labels)
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
