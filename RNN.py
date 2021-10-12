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
### ESEGUO RETE RNN  PER TRE PREDIZIONI PER FARE PLOT DI ACCURACY E LOSS
### SUL TRAINSET E VALIDSET PER 150 EPOCHE DI ADDESTRAMENTO.
### DAL PLOT RISULTA CHE L'OVERFIT INIZA FRA LA 15-ESIMA E 45-ESIMA EPOCA
### DI ADDESTRAMENTO. L'ACCURATEZZA DEL VALIDSET AUMENTA FINO A STABILIZZARSI
### VERSO LA 55-EPOCA.

### RETE CON LA MIGLIORE ACCURATEZZA SUL VALIDSET:
#    ESEGUO RETE RNN:
#    - DATASET NORMALIZZATO E 3 Labels per la prima condivisione e 4 labels per la seconda e la terza
#    - 67/70 epoche
#    - CrossEntropy
#    - 117 Batch Size per training
#    - 60 Batch Size per Validation e Test
#    - 1 livello nascosto, 531 [267] 3/4
#    - Adam con weight_decay=1e-5---> 48.39% sul valid, 51.01% sul test
#    51.01_RNN.pth

# Accuracy of the network on the 7020 test images: 51.01 %
# Accuracy of the network on the last share: 100.00 %
# Accuracy of the network on the second-last share: 81.20 %
# Accuracy of the network on the third-last share: 63.23 %

# Accuracy for class FB is: 91.4 %
# Accuracy for class FL is: 85.2 %
# Accuracy for class TW is: 80.4 %
# Accuracy for class NONE is: 52.8 %

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


input_size = 531
hidden_sizes = 267
output_size = 3

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
f1 = h5py.File('39tripleLabels.h5', 'r')

"""

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

net = NetRNN(input_size, hidden_sizes, output_size)
#net = NetMLPRNN(input_size, hidden_sizes, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), weight_decay=1e-5)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter("runs")
max = 0

for epoch in range(70):

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
net = NetRNN(input_size, hidden_sizes, output_size)
PATH = './51.01_RNN.pth'
net.load_state_dict(torch.load(PATH))

Features = f['valid/features']
Labels1 = f1['valid/labels/share1']
Labels2 = f1['valid/labels/share2']
Labels3 = f1['valid/labels/share3']

validationSet = CustomDataset(Features, Labels1, Labels2, Labels3)
validDataloader = torch.utils.data.DataLoader(validationSet, batch_size=batch_size_valid_and_test, shuffle=False)

"""

first_share = 0
second_share = 0
third_share = 0
correct = 0
total = 0
again=0
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

"""

testSet = CustomDataset(Features, Labels1, Labels2, Labels3)
testDataloader = torch.utils.data.DataLoader(testSet, batch_size=batch_size_valid_and_test, shuffle=False)


"""
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

true_matrix_1 = np.empty([0,3])
false_matrix_1 = np.empty([0,3])
true_matrix_2 = np.empty([0,4])
false_matrix_2 = np.empty([0,4])
true_matrix_3 = np.empty([0,4])
false_matrix_3 = np.empty([0,4])

sm = torch.nn.Softmax(dim=0)

with torch.no_grad():
    for data in testDataloader:
        images, labels1, labels2, labels3 = data
        output1, output2, output3 = net(images, batch_size_valid_and_test)
        # the class with the highest energy is what we choose as prediction
        _, predicted1 = torch.max(output1.data, 1)
        _, predicted2 = torch.max(output2.data, 1)
        _, predicted3 = torch.max(output3.data, 1)
        for i in range(len(predicted1)):
            if predicted1[i] == labels1[i]:
                probability= sm(output1[i])
                probability = probability.numpy()
                true_matrix_1 = np.append(true_matrix_1, [probability], axis=0)
            else:
                probability= sm(output1[i])
                probability = probability.numpy()
                false_matrix_1 = np.append(false_matrix_1, [probability], axis=0)

            if predicted2[i] == labels2[i]:
                probability= sm(output2[i])
                probability = probability.numpy()
                true_matrix_2 = np.append(true_matrix_2, [probability], axis=0)
            else:
                probability= sm(output2[i])
                probability = probability.numpy()
                false_matrix_2 = np.append(false_matrix_2, [probability], axis=0)

            if predicted3[i] == labels3[i]:
                probability= sm(output3[i])
                probability = probability.numpy()
                true_matrix_3 = np.append(true_matrix_3, [probability], axis=0)
            else:
                probability= sm(output3[i])
                probability = probability.numpy()
                false_matrix_3 = np.append(false_matrix_3, [probability], axis=0)

### show all numpy array
np.set_printoptions(threshold=np.inf)

"""
print(true_matrix_2.shape)
print(false_matrix_2.shape)
print(false_matrix_2)

print(true_matrix_1.dtype)

### True labels for sharing

Features_test = f['train/features']
Labels1_test = f1['train/labels/share1']
Labels2_test = f1['train/labels/share2']
Labels3_test = f1['train/labels/share3']

print(Labels2_test[:])

newfile = h5py.File('ProbabilityDistribution.h5', 'a')
newfile.create_dataset('share1/true', true_matrix_1.shape , dtype='float64', data=true_matrix_1)
newfile.create_dataset('share1/false', false_matrix_1.shape , dtype='float64', data=false_matrix_1)
newfile.create_dataset('share2/true', true_matrix_2.shape , dtype='float64', data=true_matrix_2)
newfile.create_dataset('share2/false', false_matrix_2.shape , dtype='float64', data=false_matrix_2)
newfile.create_dataset('share3/true', true_matrix_3.shape , dtype='float64', data=true_matrix_3)
newfile.create_dataset('share3/false', false_matrix_3.shape , dtype='float64', data=false_matrix_3)

"""

f = h5py.File('ProbabilityDistribution.h5', 'r')
x= f['share1/true']
print("x= ", x[0])
print("\n")
print("true_matrix_1= ", true_matrix_1[0])

