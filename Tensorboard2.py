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
### ESEGUO METODO IN "2share_Unrolled.py" PER FARE PLOT DI ACCURACY E LOSS
### SUL TRAINSET E VALIDSET PER 150 EPOCHE DI ADDESTRAMENTO.

### DAL PLOT RISULTA CHE L'OVERFIT INIZA FRA LA 20-ESIMA E 40-ESIMA EPOCA
### DI ADDESTRAMENTO. L'ACCURATEZZA DEL VALIDSET AUMENTA FINO A STABILIZZARSI
### VERSO LA 75-EPOCA.

### 2 METODI DIVERSI

### RETE CON LA MIGLIORE ACCURATEZZA SUL VALIDSET:
#    ESEGUO METODO 3 UNROLLED:
#    - DATASET NORMALIZZATO E 3 Labels per la prima condivisione e  4 per la seconda Labels
#    - 80/90 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 32] 3/4
#    - Adam con weight_decay=1e-5 ---> 80.51% sul valid, 81.04% sul test
#    81.04_Unrolled_Metodo3.pth

### RETE CON LA MIGLIORE ACCURATEZZA SUL VALIDSET:
#    ESEGUO METODO 2 UNROLLED:
#    - DATASET NORMALIZZATO E 3 Labels per la prima condivisione e  4 per la seconda Labels
#    - 88/150 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 32] 3/4
#    - Adam ---> 80.75% sul valid, 81.44% sul test
#    81.44_Unrolled_Metodo2.pth

# Accuracy for class FB is: 96.2 %
# Accuracy for class FL is: 93.6 %
# Accuracy for class TW is: 87.7 %
# Accuracy for class NONE is: 45.7 %


batch_size_train = 117
batch_size_valid_and_test = 60


class CustomDataset(Dataset):
    def __init__(self, Features, Labels1, Labels2, transform=None, target_transform=None):
        self.labels1 = Labels1
        self.labels2 = Labels2
        self.features = Features
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
output_size = 3


class NetMLPLatent(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.latent_size = hidden_sizes[2]
        self.fl1 = nn.Linear(input_size + self.latent_size, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fl4 = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x, h):
        x = torch.cat((x, h), 1)  # Concatenates the h tensor to input x.
        x = F.relu(self.fl1(x))
        x = F.relu(self.fl2(x))
        latent = F.relu(self.fl3(x))
        x = self.fl4(latent)
        return x, latent


class NetMLPUnrolled(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.share1 = NetMLPLatent(input_size, hidden_sizes, output_size)
        self.share2 = NetMLPLatent(input_size, hidden_sizes, output_size + 1)

    def forward(self, x, batch_size):
        share1, latent = self.share1(x, torch.zeros([batch_size, self.share1.latent_size]))
        share2, _ = self.share2(x, latent)
        return share1, share2


f = h5py.File('12LabelsNormalized.h5', 'r')
f1 = h5py.File('12doubleLabels.h5', 'r')

Features_test = f['train/features']
Labels1_test = f1['train/labels/share1']
Labels2_test = f1['train/labels/share2']

# trasform = none perché escono già come Tensori

trainingSet = CustomDataset(Features_test, Labels1_test, Labels2_test)
trainDataloader = DataLoader(trainingSet, batch_size=batch_size_train, shuffle=True)

trainingSet1 = CustomDataset(Features_test, Labels1_test, Labels2_test)
trainDataloader1 = DataLoader(trainingSet1, batch_size=batch_size_train, shuffle=False)

Features = f['valid/features']
Labels1 = f1['valid/labels/share1']
Labels2 = f1['valid/labels/share2']

validationSet = CustomDataset(Features, Labels1, Labels2)
validDataloader = torch.utils.data.DataLoader(validationSet, batch_size=batch_size_valid_and_test, shuffle=False)

net = NetMLPUnrolled(input_size, hidden_sizes, output_size)
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(net.share1.parameters())
# weight_decay=1e-5
optimizer2 = optim.Adam(net.share2.parameters())

# Writer will output to ./runs/ directory by default
writer = SummaryWriter("runs")
max = 0

for epoch in range(150):  # loop over the dataset multiple times

    print('Running Epoch: ', epoch)

    for i, data in enumerate(trainDataloader, 0):
        inputs, labels1, labels2 = data


        ### METODO 2
        
        output1, output2 = net(inputs, batch_size_train)

        optimizer1.zero_grad()
        loss1 = criterion(output1, labels1)
        loss1.backward(retain_graph=True)
        optimizer2.zero_grad()
        loss2 = criterion(output2, labels2)
        loss2.backward()
        optimizer1.step()
        optimizer2.step()

        """

        ### METODO 3

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        output1, output2 = net(inputs, batch_size_train)
        loss1 = criterion(output1, labels1)
        loss2 = criterion(output2, labels2)
        loss = loss1 + loss2
        loss.backward()

        optimizer1.step()
        optimizer2.step() 
        
        """



    running_loss_train = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in trainDataloader1:
            images, labels1, labels2 = data
            output1, output2 = net(images, batch_size_train)

            # Running_Loss_Train
            loss1 = criterion(output1, labels1)
            loss2 = criterion(output2, labels2)
            loss = loss1 + loss2
            running_loss_train += loss.item()

            _, predicted1 = torch.max(output1.data, 1)
            _, predicted2 = torch.max(output2.data, 1)
            total += labels1.size(0)
            for i in range(len(predicted1)):
                if predicted1[i] == labels1[i] and predicted2[i] == labels2[i]:
                    correct += 1

    Accuracy_Train = 100 * correct / total
    running_loss_train = running_loss_train/ 180

    running_loss_valid = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in validDataloader:
            images, labels1, labels2 = data
            output1, output2 = net(images, batch_size_valid_and_test)

            # Running_Loss_Valid
            loss1 = criterion(output1, labels1)
            loss2 = criterion(output2, labels2)
            loss = loss1 + loss2
            running_loss_valid += loss.item()

            _, predicted1 = torch.max(output1.data, 1)
            _, predicted2 = torch.max(output2.data, 1)
            total += labels1.size(0)
            for i in range(len(predicted1)):
                if predicted1[i] == labels1[i] and predicted2[i] == labels2[i]:
                    correct += 1

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

# DA TERMINALE IN BASSO -> tensorboard --logdir=runs
