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
### ESEGUO METODO DIRETTO SU 39 CLASSI PER FARE PLOT DI ACCURACY E LOSS
### SUL TRAINSET E VALIDSET PER 150 EPOCHE DI ADDESTRAMENTO.

### DAL PLOT RISULTA CHE L'OVERFIT INIZA FRA LA 20-ESIMA E 40-ESIMA EPOCA
### DI ADDESTRAMENTO. L'ACCURATEZZA DEL VALIDSET AUMENTA FINO A STABILIZZARSI
### VERSO LA 80-EPOCA.

#    ESEGUO METODO DIRECT39:
#    - DATASET NORMALIZZATO CON 2 FEATURES E 39 LABELS
#    - /80 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 2 livelli nascosti, 521 [256, 128] 39
#    - Adam --->  33.33%  sul valid, 34.67%  sul test
#    34.67_D39.pth

### RETE CON LA MIGLIORE ACCURATEZZA SUL VALIDSET:
#    ESEGUO METODO DIRECT39:
#    - DATASET NORMALIZZATO E 39 LABELS
#    - 149/150 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 2 livelli nascosti, 531 [256, 128] 39
#    - Adam --->  50.66% sul valid, 51.18% sul test
#    51Direct39.pth


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

input_size = 521 #531
hidden_sizes = [256, 128]  # [256, 128, 64]
output_size = 39

class NetMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.fl1 = nn.Linear(input_size, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        #self.fl3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fl4 = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        x = F.relu(self.fl1(x))
        x = F.relu(self.fl2(x))
        #x = F.relu(self.fl3(x))
        x = self.fl4(x)
        return x

#f = h5py.File('12LabelsNormalized.h5', 'r')
f = h5py.File('2FeaturesNormalized.h5','r')
f1 = h5py.File('39Labels.h5', 'r')

"""

Features_test = f['train/features']
Labels_test = f1['train/labels']

# trasform = none perché escono già come Tensori

trainingSet = CustomDataset(Features_test, Labels_test)
trainDataloader = DataLoader(trainingSet, batch_size=batch_size_train, shuffle=True)

trainingSet1 = CustomDataset(Features_test, Labels_test)
trainDataloader1 = DataLoader(trainingSet1, batch_size=batch_size_train, shuffle=False)

Features = f['valid/features']
Labels1 = f1['valid/labels']

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
PATH = './last.pth'
net.load_state_dict(torch.load(PATH))


Features = f['valid/features']
Labels= f1['valid/labels']

validationSet = CustomDataset(Features, Labels)
validDataloader = torch.utils.data.DataLoader(validationSet, batch_size=60, shuffle=False)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in validDataloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 7020 validation images: %.2f %%' % (100 * correct / total))

Features = f['test/features']
Labels= f1['test/labels']

testSet = CustomDataset(Features, Labels)
testDataloader = torch.utils.data.DataLoader(testSet, batch_size=60, shuffle=False)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testDataloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 7020 test images: %.2f %%' % (100 * correct / total))


