import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix

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


input_size = 531 # 521
hidden_sizes = 267 #[256, 128, 32]
output_size = 3

"""
class NetRNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.latent_size = hidden_sizes[2]
        self.fl1 = nn.Linear(input_size + self.latent_size, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fl4 = nn.Linear(hidden_sizes[2], output_size)
        self.fl5 = nn.Linear(hidden_sizes[2], output_size+1)

    def forward(self, x, batch_size):
        latent = torch.zeros([batch_size, self.latent_size])
        share = []
        for i in range(3):
            y = torch.cat((x, latent), 1)
            y = F.relu(self.fl1(y))
            y = F.relu(self.fl2(y))
            latent = F.relu(self.fl3(y))
            if i == 0:
                y = self.fl4(latent)
            else:
                y = self.fl5(latent)
            share.append(y)
        return share[0], share[1], share[2]
"""

class NetRNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.latent_size = hidden_sizes
        self.fl1 = nn.Linear(input_size + self.latent_size, hidden_sizes)
        self.fl2 = nn.Linear(hidden_sizes, output_size)
        self.fl3 = nn.Linear(hidden_sizes, output_size+1)

    def forward(self, x, batch_size):
        latent = torch.zeros([batch_size, self.latent_size])
        share = []
        for i in range(3):
            y = torch.cat((x, latent), 1)
            latent = F.relu(self.fl1(y))
            if i == 0:
                y = self.fl2(latent)
            else:
                y = self.fl3(latent)
            share.append(y)
        return share[0], share[1], share[2]

f = h5py.File('12LabelsNormalized.h5', 'r')
#f = h5py.File('2FeaturesNormalized.h5', 'r')
f1 = h5py.File('39tripleLabels.h5', 'r')

# Salvataggio
net = NetRNN(input_size, hidden_sizes, output_size)
PATH = './51.01_RNN.pth'
net.load_state_dict(torch.load(PATH))

Features = f['test/features']
Labels1 = f1['test/labels/share1']
Labels2 = f1['test/labels/share2']
Labels3 = f1['test/labels/share3']

testSet = CustomDataset(Features, Labels1, Labels2, Labels3)
testDataloader = torch.utils.data.DataLoader(testSet, batch_size=batch_size_valid_and_test, shuffle=False)

true1 = torch.empty(0)
true2 = torch.empty(0)
true3 = torch.empty(0)
pred1 = torch.empty(0)
pred2 = torch.empty(0)
pred3 = torch.empty(0)

with torch.no_grad():
    for data in testDataloader:
        images, labels1, labels2, labels3 = data
        output1, output2, output3 = net(images, batch_size_valid_and_test)
        # the class with the highest energy is what we choose as prediction
        _, predicted1 = torch.max(output1.data, 1)
        _, predicted2 = torch.max(output2.data, 1)
        _, predicted3 = torch.max(output3.data, 1)
        true1= torch.hstack((true1,labels1))
        true2= torch.hstack((true2,labels2))
        true3= torch.hstack((true3,labels3))
        pred1= torch.hstack((pred1,predicted1))
        pred2= torch.hstack((pred2,predicted2))
        pred3= torch.hstack((pred3,predicted3))

print("Confusion Matrix first level: ")
print(confusion_matrix(true1, pred1))
print("\n")
print("Confusion Matrix second level: ")
print(confusion_matrix(true2, pred2))
print("\n")
print("Confusion Matrix third level: ")
print(confusion_matrix(true3, pred3))

##########################################################
### CONFUSION MATRIX PER LIVELLO CON 50.66_RNN.pth ###
"""
Confusion Matrix first level: 
[[2340    0    0]
 [   0 2340    0]
 [   0    0 2340]]


Confusion Matrix second level: 
[[1976   40  131   13]
 [ 163 1867  105   25]
 [ 291  252 1584   33]
 [  22   76  172  270]]


Confusion Matrix third level: 
[[1112  125   67  316]
 [ 205 1013   91  311]
 [ 147  170  897  406]
 [ 334  326  152 1348]]
"""

##########################################################
### CONFUSION MATRIX PER LIVELLO CON 34.30_RNN2.pth ###

"""
Confusion Matrix first level: 
[[1948    2  390]
 [   1 2339    0]
 [ 105    0 2235]]


Confusion Matrix second level: 
[[1357  444  299   60]
 [ 140 1813  169   38]
 [ 270  493 1369   28]
 [  90  137  111  202]]


Confusion Matrix third level: 
[[ 631  407   94  488]
 [ 245 1028   73  274]
 [ 191  193  885  351]
 [ 417  450  200 1093]]
"""

"""

FBFB = 0
FBFL = 0
FBTW = 0
FLFB = 0
FLFL = 0
FLTW = 0
TWFB = 0
TWFL = 0
TWTW = 0

with torch.no_grad():
    for data in testDataloader:
        images, labels1, labels2, labels3 = data
        output1, output2, output3 = net(images, batch_size_valid_and_test)
        _, predicted1 = torch.max(output1.data, 1)
        _, predicted2 = torch.max(output2.data, 1)
        _, predicted3 = torch.max(output3.data, 1)
        for i, j, k, a, b in zip(labels1, labels2, labels3, predicted1, predicted2):
            if j != 3. and k == 3.:
               if i == 0. and j == 0.:
                   if a == 0. and b == 0.:
                        FBFB +=1
               if i == 0. and j == 1.:
                   if a == 0. and b == 1.:
                        FBFL +=1
               if i == 0. and j == 2.:
                   if a == 0. and b == 2.:
                        FBTW +=1
               if i == 1. and j == 0.:
                   if a == 1. and b == 0.:
                        FLFB +=1
               if i == 1. and j == 1.:
                   if a == 1. and b == 1.:
                        FLFL +=1
               if i == 1. and j == 2.:
                   if a == 1. and b == 2.:
                        FLTW +=1
               if i == 2. and j == 0.:
                   if a == 2. and b == 0.:
                        TWFB +=1
               if i == 2. and j == 1.:
                   if a == 2. and b == 1.:
                        TWFL +=1
               if i == 2. and j == 2.:
                   if a == 2. and b == 2.:
                        TWTW +=1

ConfusionMatrix = torch.empty((0,3), dtype=torch.int)

FBFB = torch.tensor(FBFB, dtype=torch.int)
FBFL = torch.tensor(FBFL, dtype=torch.int)
FBTW = torch.tensor(FBTW, dtype=torch.int)
FLFB = torch.tensor(FLFB, dtype=torch.int)
FLFL = torch.tensor(FLFL, dtype=torch.int)
FLTW = torch.tensor(FLTW, dtype=torch.int)
TWFB = torch.tensor(TWFB, dtype=torch.int)
TWFL = torch.tensor(TWFL, dtype=torch.int)
TWTW = torch.tensor(TWTW, dtype=torch.int)

row = torch.empty(0, dtype=torch.int)
row = torch.hstack((row, FBFB))
row = torch.hstack((row, FBFL))
row = torch.hstack((row, FBTW))

ConfusionMatrix = torch.vstack((ConfusionMatrix,row))

row = torch.empty(0, dtype=torch.int)
row = torch.hstack((row, FLFB))
row = torch.hstack((row, FLFL))
row = torch.hstack((row, FLTW))

ConfusionMatrix = torch.vstack((ConfusionMatrix,row))

row = torch.empty(0, dtype=torch.int)
row = torch.hstack((row, TWFB))
row = torch.hstack((row, TWFL))
row = torch.hstack((row, TWTW))

ConfusionMatrix = torch.vstack((ConfusionMatrix,row))

print("Confusion Matrix for last and second last share: ")
print(ConfusionMatrix)
"""

######################################################
### Matrice di Confusione per ultima e penultima condivisione su 2 share con 50.66_RNN.pth ###

"""
    FB   FL   TW
FB [163, 150, 172]
FL [167, 177, 166]
TW [170, 128, 120]

"""

###############################################################
### CONFUSION MATRIX PER LIVELLO CON 51.01_RNN.pth ###
"""

Confusion Matrix first level: 
[[2340    0    0]
 [   0 2340    0]
 [   0    0 2340]]


Confusion Matrix second level: 
[[1961   61  129    9]
 [ 148 1841  136   35]
 [ 281  214 1655   10]
 [  19   70  208  243]]


Confusion Matrix third level: 
[[1293   94   80  153]
 [ 312 1036  100  172]
 [ 218  174  927  301]
 [ 489  316  172 1183]]

"""
