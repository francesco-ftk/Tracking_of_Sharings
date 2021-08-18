import torch
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


######################################################################################
#    ESEGUO METODO UNROLLED:
#    - DATASET NORMALIZZATO E 4 Labels
#    - 35 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 16] 4
#    - Adam ---> 79%
#    79UnrolledAdam.pth

#    ESEGUO METODO UNROLLED:
#    - DATASET NORMALIZZATO E 3 Labels per la prima condivisione e  4 per la seconda Labels
#    - 80 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 32] 3/4
#    - Adam ---> 79.74% sul valid, 81.14% sul test
#    79e81UnrolledAdam.pth

#    ESEGUO METODO UNROLLED:
#    - DATASET NORMALIZZATO E 4 Labels
#    - 45 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 16] 4
#    - optimizer SGD con Nesterov Momentum ---> 75%
#    NesterovUnrolled75.pth

batch_size_train = 117
batch_size_valid_and_test = 60

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
        x = torch.cat((x, h), 1)   # Concatenates the h tensor to input x.
        x = F.relu(self.fl1(x))
        x = F.relu(self.fl2(x))
        latent = F.relu(self.fl3(x))
        x = self.fl4(latent)
        return x, latent

class NetMLPUnrolled(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.share1 = NetMLPLatent(input_size, hidden_sizes, output_size)
        self.share2 = NetMLPLatent(input_size, hidden_sizes, output_size+1)

    def forward(self, x, batch_size):
        share1, latent = self.share1(x, torch.zeros([batch_size, self.share1.latent_size]))
        share2, _ = self.share2(x, latent)
        return share1, share2

f = h5py.File('12LabelsNormalized.h5', 'r')
f1 = h5py.File('doubleLabels.h5', 'r')

"""

Features = f['train/features']
Labels1 = f1['train/labels/share1']
Labels2 = f1['train/labels/share2']

# trasform = none perché escono già come Tensori

trainingSet = CustomDataset(Features, Labels1, Labels2)
trainDataloader = DataLoader(trainingSet, batch_size=batch_size_train, shuffle=True)

net = NetMLPUnrolled(input_size, hidden_sizes, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)

for epoch in range(80):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainDataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels1, labels2 = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output1, output2 = net(inputs, batch_size_train)
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
net = NetMLPUnrolled(input_size, hidden_sizes, output_size)
PATH = './79e81UnrolledAdam.pth'
net.load_state_dict(torch.load(PATH))

Features = f['valid/features']
Labels1 = f1['valid/labels/share1']
Labels2 = f1['valid/labels/share2']

validationSet = CustomDataset(Features, Labels1, Labels2)
validDataloader = torch.utils.data.DataLoader(validationSet, batch_size=batch_size_valid_and_test, shuffle=False)

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
        output1, output2 = net(images,batch_size_valid_and_test)
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
        output1, output2 = net(images,batch_size_valid_and_test)
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

print('Accuracy of the network on the 7020 test images: %.2f %%' % (100 * correct / total))
print('Accuracy of the network on the last share: %.2f %%' % (100 * first_share/ total))
print('Accuracy of the network on the second-last share: %.2f %%' % (100 * second_share/ total))


"""

class NetJoin(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.latent_size = hidden_sizes[2]
        self.fl1 = nn.Linear(input_size + self.latent_size, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        #self.fl4 = nn.Linear(hidden_sizes[2], output_size)

         # Classifier for last share.
        self.share1 = nn.Linear(hidden_sizes[2], output_size)
        # Classifier for penultimate share.
        self.share2 = nn.Linear(hidden_sizes[2], output_size+1)

    def forward(self, x, batch_size):
        h = torch.zeros([batch_size,self.latent_size])
        share1 = torch.cat((x, h), 1)   # Concatenates the h tensor to input x.
        share1 = F.relu(self.fl1(share1))
        share1 = F.relu(self.fl2(share1))
        latent = F.relu(self.fl3(share1))
        share1 = self.share1(latent)

        share2 = torch.cat((x, latent), 1)   # Concatenates the h tensor to input x.
        share2 = F.relu(self.fl1(share2))
        share2 = F.relu(self.fl2(share2))
        share2 = F.relu(self.fl3(share2))
        share2 = self.share2(share2)
        return share1, share2   
"""


