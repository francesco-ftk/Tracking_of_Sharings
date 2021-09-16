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
### ESEGUO METODO UNROLLED PER TRE PREDIZIONI PER FARE PLOT DI ACCURACY E LOSS
### SUL TRAINSET E VALIDSET PER 150 EPOCHE DI ADDESTRAMENTO.
### DAL PLOT RISULTA CHE L'OVERFIT INIZA FRA LA 20-ESIMA E 40-ESIMA EPOCA
### DI ADDESTRAMENTO. L'ACCURATEZZA DEL VALIDSET AUMENTA FINO A STABILIZZARSI
### VERSO LA 55-EPOCA.

### RETE CON LA MIGLIORE ACCURATEZZA SUL VALIDSET:
#    ESEGUO METODO UNROLLED:
#    - DATASET NORMALIZZATO E 3 Labels per la prima condivisione e 4 per la seconda e la terza Labels
#    - 80/90 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 521 [256, 128, 32] 3/4
#    - Adam con weight_decay=1e-5---> 48.45% sul valid, 50.24% sul test
#    50.24_Unrolled3.pth

# Accuracy of the network on the 7020 test images: 50.24 %
# Accuracy of the network on the last share: 100.00 %
# Accuracy of the network on the second-last share: 81.64 %
# Accuracy of the network on the third-last share: 62.86 %

# Accuracy for class FB is: 88.0 %
# Accuracy for class FL is: 86.5 %
# Accuracy for class TW is: 80.0 %
# Accuracy for class NONE is: 58.7 %

### RETE CON LA MIGLIORE ACCURATEZZA SUL VALIDSET:
#    ESEGUO METODO UNROLLED:
#    - DATASET NORMALIZZATO con 2 features E 3 Labels per la prima condivisione e 4 per la seconda e la terza Labels
#    - 80 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 32] 3/4
#    - Adam con weight_decay=1e-5---> 31.99% sul valid, 33.86% sul test
#    33.86_Unrolled3.pth

# Accuracy of the network on the 7020 test images: 33.86 %
# Accuracy of the network on the last share: 92.92 %
# Accuracy of the network on the second-last share: 67.82 %
# Accuracy of the network on the third-last share: 51.95 %

# Accuracy for class FB is: 68.8 %
# Accuracy for class FL is: 81.6 %
# Accuracy for class TW is: 73.2 %
# Accuracy for class NONE is: 46.3 %


batch_size_train = 117
batch_size_valid_and_test = 60


class CustomDataset(Dataset):
    def __init__(self, Features, Labels1, Labels2, Labels3, transform=None, target_transform=None):
        self.labels1 = Labels1
        self.labels2 = Labels2
        self.labels3 = Labels3
        self.features = Features
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels1)

    def __getitem__(self, idx):
        image = self.features[idx]
        labels1 = self.labels1[idx]
        labels2 = self.labels2[idx]
        labels3 = self.labels3[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels1 = self.target_transform(labels1)
            labels2 = self.target_transform(labels2)
            labels3 = self.target_transform(labels3)
        return image, labels1, labels2, labels3


input_size = 521  # 531
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
        self.share3 = NetMLPLatent(input_size, hidden_sizes, output_size + 1)

    def forward(self, x, batch_size):
        share1, latent = self.share1(x, torch.zeros([batch_size, self.share1.latent_size]))
        share2, latent = self.share2(x, latent)
        share3, _ = self.share2(x, latent)
        return share1, share2, share3


f = h5py.File('2FeaturesNormalized.h5','r')
#f = h5py.File('12LabelsNormalized.h5', 'r')
f1 = h5py.File('39tripleLabels.h5', 'r')

Features_test = f['train/features']
Labels1_test = f1['train/labels/share1']
Labels2_test = f1['train/labels/share2']
Labels3_test = f1['train/labels/share3']

# trasform = none perché escono già come Tensori

trainingSet = CustomDataset(Features_test, Labels1_test, Labels2_test, Labels3_test)
trainDataloader = DataLoader(trainingSet, batch_size=batch_size_train, shuffle=True)

trainingSet1 = CustomDataset(Features_test, Labels1_test, Labels2_test, Labels3_test)
trainDataloader1 = DataLoader(trainingSet1, batch_size=batch_size_train, shuffle=False)

Features = f['valid/features']
Labels1 = f1['valid/labels/share1']
Labels2 = f1['valid/labels/share2']
Labels3 = f1['valid/labels/share3']

validationSet = CustomDataset(Features, Labels1, Labels2, Labels3)
validDataloader = torch.utils.data.DataLoader(validationSet, batch_size=batch_size_valid_and_test, shuffle=False)

net = NetMLPUnrolled(input_size, hidden_sizes, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), weight_decay=1e-5)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter("runs")
max = 0

for epoch in range(80):  # loop over the dataset multiple times

    print('Running Epoch: ', epoch)

    for i, data in enumerate(trainDataloader, 0):
        inputs, labels1, labels2, labels3 = data

        optimizer.zero_grad()

        output1, output2, output3 = net(inputs, batch_size_train)
        loss1 = criterion(output1, labels1)
        loss2 = criterion(output2, labels2)
        loss3 = criterion(output3, labels3)
        loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()

    running_loss_train = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in trainDataloader1:
            images, labels1, labels2, labels3 = data
            output1, output2, output3 = net(images, batch_size_train)

            # Running_Loss_Train
            loss1 = criterion(output1, labels1)
            loss2 = criterion(output2, labels2)
            loss3 = criterion(output3, labels3)
            loss = loss1 + loss2 + loss3
            running_loss_train += loss.item()

            _, predicted1 = torch.max(output1.data, 1)
            _, predicted2 = torch.max(output2.data, 1)
            _, predicted3 = torch.max(output3.data, 1)
            total += labels1.size(0)
            for i in range(len(predicted1)):
                if predicted1[i] == labels1[i] and predicted2[i] == labels2[i] and predicted3[i] == labels3[i]:
                    correct += 1

    Accuracy_Train = 100 * correct / total
    running_loss_train = running_loss_train/ 180

    running_loss_valid = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in validDataloader:
            images, labels1, labels2, labels3 = data
            output1, output2, output3 = net(images, batch_size_valid_and_test)

            # Running_Loss_Valid
            loss1 = criterion(output1, labels1)
            loss2 = criterion(output2, labels2)
            loss3 = criterion(output3, labels3)
            loss = loss1 + loss2 + loss3
            running_loss_valid += loss.item()

            _, predicted1 = torch.max(output1.data, 1)
            _, predicted2 = torch.max(output2.data, 1)
            _, predicted3 = torch.max(output3.data, 1)
            total += labels1.size(0)
            for i in range(len(predicted1)):
                if predicted1[i] == labels1[i] and predicted2[i] == labels2[i] and predicted3[i] == labels3[i]:
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

"""

# Salvataggio
net = NetMLPUnrolled(input_size, hidden_sizes, output_size)
PATH = './last.pth'
net.load_state_dict(torch.load(PATH))

Features = f['valid/features']
Labels1 = f1['valid/labels/share1']
Labels2 = f1['valid/labels/share2']
Labels3 = f1['valid/labels/share3']

validationSet = CustomDataset(Features, Labels1, Labels2, Labels3)
validDataloader = torch.utils.data.DataLoader(validationSet, batch_size=batch_size_valid_and_test, shuffle=False)

first_share = 0
second_share = 0
third_share = 0
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in validDataloader:
        images, labels1, labels2, labels3 = data
        output1, output2, output3 = net(images, batch_size_valid_and_test)
        # the class with the highest energy is what we choose as prediction
        _, predicted1 = torch.max(output1.data, 1)
        _, predicted2 = torch.max(output2.data, 1)
        _, predicted3 = torch.max(output3.data, 1)
        total += labels1.size(0)
        for i in range(len(predicted1)):
            if predicted1[i] == labels1[i]:
                first_share += 1
            if predicted2[i] == labels2[i]:
                second_share += 1
            if predicted3[i] == labels3[i]:
                third_share += 1
            if predicted1[i] == labels1[i] and predicted2[i] == labels2[i] and predicted3[i] == labels3[i]:
                correct += 1

print('Accuracy of the network on the 7020 validation images: %.2f %%' % (100 * correct / total))
print('Accuracy of the network on the last share: %.2f %%' % (100 * first_share / total))
print('Accuracy of the network on the second-last share: %.2f %%' % (100 * second_share / total))
print('Accuracy of the network on the third-last share: %.2f %%' % (100 * third_share / total))

Features = f['test/features']
Labels1 = f1['test/labels/share1']
Labels2 = f1['test/labels/share2']
Labels3 = f1['test/labels/share3']

testSet = CustomDataset(Features, Labels1, Labels2, Labels3)
testDataloader = torch.utils.data.DataLoader(testSet, batch_size=60, shuffle=False)

first_share = 0
second_share = 0
third_share = 0
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testDataloader:
        images, labels1, labels2, labels3 = data
        output1, output2, output3 = net(images, batch_size_valid_and_test)
        # the class with the highest energy is what we choose as prediction
        _, predicted1 = torch.max(output1.data, 1)
        _, predicted2 = torch.max(output2.data, 1)
        _, predicted3 = torch.max(output3.data, 1)
        total += labels1.size(0)
        for i in range(len(predicted1)):
            if predicted1[i] == labels1[i]:
                first_share += 1
            if predicted2[i] == labels2[i]:
                second_share += 1
            if predicted3[i] == labels3[i]:
                third_share += 1
            if predicted1[i] == labels1[i] and predicted2[i] == labels2[i] and predicted3[i] == labels3[i]:
                correct += 1

print("\n")
print('Accuracy of the network on the 7020 test images: %.2f %%' % (100 * correct / total))
print('Accuracy of the network on the last share: %.2f %%' % (100 * first_share / total))
print('Accuracy of the network on the second-last share: %.2f %%' % (100 * second_share / total))
print('Accuracy of the network on the third-last share: %.2f %%' % (100 * third_share / total))

# STAMPO ACCURATEZZA PER OGNI CLASSE SUL TESTSET

classes = ('FB', 'FL', 'TW', 'NONE')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testDataloader:
        images, labels1, labels2, labels3 = data
        output1, output2, output3 = net(images, batch_size_valid_and_test)
        _, predicted1 = torch.max(output1.data, 1)
        _, predicted2 = torch.max(output2.data, 1)
        _, predicted3 = torch.max(output3.data, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels1, predicted1):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1
        for label, prediction in zip(labels2, predicted2):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1
        for label, prediction in zip(labels3, predicted3):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
print("\n")
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:2s} is: {:.1f} %".format(classname,
                                                         accuracy))
"""
