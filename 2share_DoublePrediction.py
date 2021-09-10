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
#    ESEGUO METODO MLP CON DOPPIO OUTPUT CONTEMPORANEO E LOSS SOMMATA CON:
#    - DATASET NORMALIZZATO E 3 Labels per la prima share e 4 Labels per la seconda
#    - 60 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 32] 3/4
#    - optimizer SGD con Nesterov Momentum ---> 75% , 77% nel test
#    NesterovDoubleSum75.pth

#    ESEGUO METODO MLP CON DOPPIO OUTPUT CONTEMPORANEO E LOSS SOMMATA CON:
#    - DATASET NORMALIZZATO E 3 Labels per la prima share e 4 Labels per la seconda
#    - 25 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 2 livelli nascosti, 531 [128, 32] 3/4
#    - optimizer Adam ---> 78%, 79% nel test
#    Adam2HideDoubleSum78.pth

#    ESEGUO METODO MLP CON DOPPIO OUTPUT CONTEMPORANEO E LOSS SOMMATA CON:
#    - DATASET NORMALIZZATO E 3 Labels per la prima share e 4 Labels per la seconda
#    - 71/80 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 32] 3/4
#    - optimizer Adam ---> 80%, 81.08% nel test
#    81.08_DoublePrediction.pth

#Accuracy of the network on the 7020 test images: 81.08 %
#Accuracy of the network on the last share: 100.00 %
#Accuracy of the network on the second-last share: 81.08 %

#Accuracy for class FB is: 95.1 %
#Accuracy for class FL is: 92.6 %
#Accuracy for class TW is: 89.3 %
#Accuracy for class NONE is: 45.7 %

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
output_size1 = 3
output_size2 = 4


class NetMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size1, output_size2):
        super().__init__()
        self.fl1 = nn.Linear(input_size, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])

        # Classifier for last share.
        self.share1 = nn.Linear(hidden_sizes[2], output_size1)
        # Classifier for penultimate share.
        self.share2 = nn.Linear(hidden_sizes[2], output_size2)

    def forward(self, x):
        x = F.relu(self.fl1(x))
        x = F.relu(self.fl2(x))
        x = F.relu(self.fl3(x))
        # Compute outputs.
        share1 = self.share1(x)
        share2 = self.share2(x)
        return share1, share2

f = h5py.File('12LabelsNormalized.h5', 'r')
f1 = h5py.File('12doubleLabels.h5', 'r')

"""

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

net = NetMLP(input_size, hidden_sizes, output_size1, output_size2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter("runs")
max = 0

for epoch in range(80):  # loop over the dataset multiple times

    print('Running Epoch: ', epoch)

    for i, data in enumerate(trainDataloader, 0):
        inputs, labels1, labels2 = data

        # zero the parameter gradients
        optimizer.zero_grad()

        output1, output2 = net(inputs)
        loss1 = criterion(output1, labels1)
        loss2 = criterion(output2, labels2)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

    running_loss_train = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in trainDataloader1:
            images, labels1, labels2 = data
            output1, output2 = net(images)

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
            output1, output2 = net(images)

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

"""

# Salvataggio
net = NetMLP(input_size, hidden_sizes, output_size1, output_size2)
PATH = './last.pth'
net.load_state_dict(torch.load(PATH))

Features = f['valid/features']
Labels1 = f1['valid/labels/share1']
Labels2 = f1['valid/labels/share2']

validationSet = CustomDataset(Features, Labels1, Labels2)
validDataloader = torch.utils.data.DataLoader(validationSet, batch_size=60, shuffle=False)

first_share = 0
second_share = 0
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
            if predicted1[i] == labels1[i]:
                first_share +=1
                if predicted2[i] == labels2[i]:
                    second_share +=1
                    correct +=1
            elif predicted2[i] == labels2[i]:
                    second_share +=1

print('Accuracy of the network on the 7020 validation images: %.2f %%' % (100 * correct / total))
print('Accuracy of the network on the last share: %.2f %%' % (100 * first_share/ total))
print('Accuracy of the network on the second-last share: %.2f %%' % (100 * second_share/ total))

Features = f['test/features']
Labels1 = f1['test/labels/share1']
Labels2 = f1['test/labels/share2']

testSet = CustomDataset(Features, Labels1, Labels2)
testDataloader = torch.utils.data.DataLoader(testSet, batch_size=60, shuffle=False)

first_share = 0
second_share = 0
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
            if predicted1[i] == labels1[i]:
                first_share +=1
                if predicted2[i] == labels2[i]:
                    second_share +=1
                    correct +=1
            elif predicted2[i] == labels2[i]:
                    second_share +=1

print("\n")
print('Accuracy of the network on the 7020 test images: %.2f %%' % (100 * correct / total))
print('Accuracy of the network on the last share: %.2f %%' % (100 * first_share/ total))
print('Accuracy of the network on the second-last share: %.2f %%' % (100 * second_share/ total))

# STAMPO ACCURATEZZA PER OGNI CLASSE SUL TESTSET

classes = ('FB', 'FL', 'TW', 'NONE')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testDataloader:
        images, labels1, labels2 = data
        output1, output2 = net(images)
        _, predicted1 = torch.max(output1.data, 1)
        _, predicted2 = torch.max(output2.data, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels1, predicted1):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1
        for label, prediction in zip(labels2, predicted2):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
print("\n")
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:2s} is: {:.1f} %".format(classname,
                                                   accuracy))


