import torch
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

######################################################################################
#    ESEGUO METODO MLP CON DOPPIO OUTPUT CONTEMPORANEO E LOSS SOMMATA CON:
#    - DATASET NORMALIZZATO E 3 Labels per la prima share e 4 Labels per la seconda
#    - 60 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 32] 3/4
#    - optimizer SGD con Nesterov Momentum ---> 75%
#    NesterovDoubleSum75.pth

#    ESEGUO METODO MLP CON DOPPIO OUTPUT CONTEMPORANEO E LOSS SOMMATA CON:
#    - DATASET NORMALIZZATO E 3 Labels per la prima share e 4 Labels per la seconda
#    - 45 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 32] 3/4
#    - optimizer Adam ---> 77%
#    AdamDoubleSum77.pth

#    ESEGUO METODO MLP CON DOPPIO OUTPUT CONTEMPORANEO E LOSS SOMMATA CON:
#    - DATASET NORMALIZZATO E 3 Labels per la prima share e 4 Labels per la seconda
#    - 25 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 2 livelli nascosti, 531 [128, 32] 3/4
#    - optimizer Adam ---> 78%
#    Adam2HideDoubleSum78.pth

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

#classes = ('FB', 'FL', 'TW', 'none')

f = h5py.File('12LabelsNormalized.h5', 'r')
f1 = h5py.File('doubleLabels.h5', 'r')


Features = f['train/features']
Labels1 = f1['train/labels/share1']
Labels2 = f1['train/labels/share2']

# trasform = none perché escono già come Tensori

trainingSet = CustomDataset(Features, Labels1, Labels2)
trainDataloader = DataLoader(trainingSet, batch_size=117, shuffle=True)

net = NetMLP(input_size, hidden_sizes, output_size1, output_size2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)

"""

dataiter = iter(trainDataloader)
images, labels = dataiter.next()
print(labels.shape)
print(labels)
print(labels[0])
print(labels[0,0])
print(labels[0,1])
print(labels[1,0])
print(labels[1,1])

dataiter = iter(trainDataloader)
images, labels1, labels2 = dataiter.next()
print(labels1.shape)
print(labels1)
print(labels2.shape)
print(labels2)
print(images.shape)

"""

for epoch in range(25):  # loop over the dataset multiple times

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

# Salvataggio
#net = NetMLP(input_size, hidden_sizes, output_size1, output_size2)
#PATH = './last.pth'
#net.load_state_dict(torch.load(PATH))

Features = f['valid/features']
Labels1 = f1['valid/labels/share1']
Labels2 = f1['valid/labels/share2']

validationSet = CustomDataset(Features,Labels1,Labels2)
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
                correct+=1
                #print("total: ", total)
                #print("correct: ", correct)


print('Accuracy of the network on the 7020 validation images: %d %%' % (100 * correct / total))


"""

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

print('Accuracy of the network on the 7020 test images: %d %%' % (100 * correct / total))

"""
