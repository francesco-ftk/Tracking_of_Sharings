import torch
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


"""

######################################################################################
 ### CODICE PER CREARE UN DATASET CON 3 CLASSI ###

f = h5py.File('3Labels.h5', 'a')

# Apro il file h5py
file = h5py.File('dataset.h5', 'r')
# Stampo le chiavi dei primi gruppi presenti (dizionari)
print(list(file.keys()))
# Prendo un gruppo tramite la sua chiave
trainSet = file['train']
testSet = file['test']
validSet = file['valid']

# TestSet e ValidationSet di 7020 immagini, 39 etichette possibili, per ognuna abbiamo 369 dct, 10 header e 152 meta, ossia 532 features
# TrainSet di 21060 immagini, 39 etichette possibili, per ognuna abbiamo 369 dct, 10 header e 152 meta, ossia 532 features

def setInput(numOfImg, dct, header, meta):
    row = np.concatenate((dct[0], header[0], meta[0]), axis=0)
    row1 = np.concatenate((dct[1], header[1], meta[1]), axis=0)
    input = np.vstack((row, row1))
    for i in range(2, numOfImg, 1):
        row = np.concatenate((dct[i], header[i], meta[i]), axis=0)
        input = np.vstack((input, row))
        if i%351 == 0:
            print(i)
    return input

def reduceTo3Labels(labels):
      newLabels= np.empty([0,0])
      for i in range(0,labels.shape[0],1):
           newLabels = np.append(newLabels,labels[i]%3)
      newLabels = np.int_(newLabels)
      return newLabels

# preparo il trainSet
trainFeatures = trainSet['features']
dct = trainFeatures['dct']
header = trainFeatures['header']
meta = trainFeatures['meta']

labels = trainSet['labels']

trainSet = setInput(labels.shape[0], dct, header, meta)
features = f.create_dataset('train/features', (21060,531), dtype='float32', data=trainSet)
trainLabels= reduceTo3Labels(labels)
labels = f.create_dataset('train/labels', (21060,), dtype='int64', data=trainLabels)

# preparo il validSet
validFeatures = validSet['features']
dct = validFeatures['dct']
header = validFeatures['header']
meta = validFeatures['meta']

labels = validSet['labels']

validSet = setInput(labels.shape[0], dct, header, meta)
features = f.create_dataset('valid/features', (7020,531), dtype='float32', data=validSet)
validLabels= reduceTo3Labels(labels)
labels = f.create_dataset('valid/labels', (7020,), dtype='int64', data=validLabels)

# preparo il testSet
testFeatures = testSet['features']
dct = testFeatures['dct']
header = testFeatures['header']
meta = testFeatures['meta']

labels = testSet['labels']

testSet = setInput(labels.shape[0], dct, header, meta)
features = f.create_dataset('test/features', (7020,531), dtype='float32', data=testSet)
testLabels= reduceTo3Labels(labels)
labels = f.create_dataset('test/labels', (7020,), dtype='int64', data=testLabels)


"""

######################################################################################
#    ESEGUO MLP CON:
#    - DATASET NON NORMALIZZATO E 3 Labels
#    - 20 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 3 livelli nascosti, 531 [256, 128, 32] 3
#    - optimizer Adam ---> 100%

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
output_size = 3

class NetMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.fl1 = nn.Linear(input_size, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fl6 = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        x = F.relu(self.fl1(x))
        x = F.relu(self.fl2(x))
        x = F.relu(self.fl3(x))
        x = self.fl6(x)
        return x

classes = ('FB', 'FL', 'TW')

f = h5py.File('3Labels.h5', 'r')

"""
trainSet = f['train']
Features = trainSet['features']
Labels= trainSet['labels']

# trasform = none perché escono già come Tensori

trainingSet = CustomDataset(Features,Labels)
trainDataloader = DataLoader(trainingSet, batch_size=117, shuffle=True)

#dataiter = iter(trainDataloader)
#images, labels = dataiter.next()

net = NetMLP(input_size, hidden_sizes, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(20):  # loop over the dataset multiple times

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
PATH = './100Adam.pth'
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

