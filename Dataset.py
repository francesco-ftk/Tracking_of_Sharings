import h5py
import numpy as np

# TestSet e ValidationSet di 7020 immagini, 39 etichette possibili, per ognuna abbiamo 369 dct, 10 header e 152 meta, ossia 532 features
# TrainSet di 21060 immagini, 39 etichette possibili, per ognuna abbiamo 369 dct, 10 header e 152 meta, ossia 532 features

######################################################################################
 ### CODICE PER CREARE UN DATASET CON FEATURES NORMALIZZATE E 3 o 12 LABELS ###

def setInput(numberOfData, dct, header, meta):
    row = np.concatenate((dct[0], header[0], meta[0]), axis=0)
    row1 = np.concatenate((dct[1], header[1], meta[1]), axis=0)
    input = np.vstack((row, row1))
    for i in range(2, numberOfData, 1):
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

def reduceTo12Labels(labels):
    newLabels = np.empty([0,0])
    for i in range(0,labels.shape[0],1):
        if(labels[i] == 12 or labels[i] == 21 or labels[i] == 30):
            newLabels = np.append(newLabels,3)
        elif(labels[i] == 13 or labels[i] == 22 or labels[i] == 31):
            newLabels = np.append(newLabels,4)
        elif(labels[i] == 14 or labels[i] == 23 or labels[i] == 32):
            newLabels = np.append(newLabels,5)
        elif(labels[i] == 15 or labels[i] == 24 or labels[i] == 33):
            newLabels = np.append(newLabels,6)
        elif(labels[i] == 16 or labels[i] == 25 or labels[i] == 34):
            newLabels = np.append(newLabels,7)
        elif(labels[i] == 17 or labels[i] == 26 or labels[i] == 35):
            newLabels = np.append(newLabels,8)
        elif(labels[i] == 18 or labels[i] == 27 or labels[i] == 36):
            newLabels = np.append(newLabels,9)
        elif(labels[i] == 19 or labels[i] == 28 or labels[i] == 37):
            newLabels = np.append(newLabels,10)
        elif(labels[i] == 20 or labels[i] == 29 or labels[i] == 38):
            newLabels = np.append(newLabels,11)
        else:
            newLabels = np.append(newLabels,labels[i])
    newLabels = np.int_(newLabels)
    return newLabels

def Normalize(file, newfile):

    dctTrain = file['train/features/dct']
    headerTrain = file['train/features/header']
    metaTrain = file['train/features/meta']

    dctValid = file['valid/features/dct']
    headerValid = file['valid/features/header']
    metaValid = file['valid/features/meta']

    dctTest = file['test/features/dct']
    headerTest = file['test/features/header']
    metaTest = file['test/features/meta']

    newDCTtrain= np.zeros((21060,369))
    newDCTvalid= np.zeros((7020,369))
    newDCTtest= np.zeros((7020,369))

    newHEADERtrain= np.zeros((21060,10))
    newHEADERvalid= np.zeros((7020,10))
    newHEADERtest= np.zeros((7020,10))

    newMETAtrain= np.zeros((21060,152))
    newMETAvalid= np.zeros((7020,152))
    newMETAtest= np.zeros((7020,152))


    for i in range(dctTrain.shape[1]):
        colTrain = dctTrain[:,i]
        colValid = dctValid[:,i]
        colTest = dctTest[:,i]
        max = np.amax(colTrain)
        min = np.amin(colTrain)
        den= max-min
        if den < np.finfo(float).eps:
            den = np.finfo(float).eps
        colTrain = (colTrain-min)/den
        colValid = (colValid-min)/den
        colTest = (colTest-min)/den
        newDCTtrain[:,i] = colTrain
        newDCTvalid[:,i] = colValid
        newDCTtest[:,i] = colTest

    for i in range(headerTrain.shape[1]):
        colTrain = headerTrain[:,i]
        colValid = headerValid[:,i]
        colTest = headerTest[:,i]
        max = np.amax(colTrain)
        min = np.amin(colTrain)
        den= max-min
        if den < np.finfo(float).eps:
            den = np.finfo(float).eps
        colTrain = (colTrain-min)/den
        colValid = (colValid-min)/den
        colTest = (colTest-min)/den
        newHEADERtrain[:,i] = colTrain
        newHEADERvalid[:,i] = colValid
        newHEADERtest[:,i] = colTest

    for i in range(metaTrain.shape[1]):
        colTrain = metaTrain[:,i]
        colValid = metaValid[:,i]
        colTest = metaTest[:,i]
        max = np.amax(colTrain)
        min = np.amin(colTrain)
        den= max-min
        if den < np.finfo(float).eps:
            den = np.finfo(float).eps
        colTrain = (colTrain-min)/den
        colValid = (colValid-min)/den
        colTest = (colTest-min)/den
        newMETAtrain[:,i] = colTrain
        newMETAvalid[:,i] = colValid
        newMETAtest[:,i] = colTest

    trainSet = setInput(21060, newDCTtrain, newHEADERtrain, newMETAtrain)
    newfile.create_dataset('train/features', (21060,531), dtype='float32', data=trainSet)
    validSet = setInput(7020, newDCTvalid, newHEADERvalid, newMETAvalid)
    newfile.create_dataset('valid/features', (7020,531), dtype='float32', data=validSet)
    testSet = setInput(7020, newDCTtest, newHEADERtest, newMETAtest)
    newfile.create_dataset('test/features', (7020,531), dtype='float32', data=testSet)

"""

# Apro il file h5py
file = h5py.File('dataset.h5', 'r')
# Stampo le chiavi dei primi gruppi presenti (dizionari)
print(list(file.keys()))
# Prendo un gruppo tramite la sua chiave
trainLabels = file['train/labels']
validLabels = file['valid/labels']
testLabels = file['test/labels']

newfile = h5py.File('12LabelsNormalized.h5', 'a')

Normalize(file, newfile)

trainLabels= reduceTo12Labels(trainLabels)
newfile.create_dataset('train/labels', (21060,), dtype='int64', data=trainLabels)
validLabels= reduceTo12Labels(validLabels)
newfile.create_dataset('valid/labels', (7020,), dtype='int64', data=validLabels)
testLabels= reduceTo12Labels(testLabels)
newfile.create_dataset('test/labels', (7020,), dtype='int64', data=testLabels)

newfile.close()
file.close()

"""

######################################################################################
 ### CODICE PER CREARE DUE LABELS: UNA SU 3 CLASSI
 ### PER LA PRIMA CONDIVISIONE (3 social) E L'ALTRA SU 4 CLASSI PER LA
 ### SECONDA (3 social + none) ###

"""
def toDoubleLabels1(labels):
    newLabels = np.empty((0,2))
    for i in range(0,labels.shape[0],1):
        if(labels[i] == 0):
            newLabels = np.vstack((newLabels, np.array([[0,4]])))
        elif(labels[i] == 1):
            newLabels = np.vstack((newLabels, np.array([[1,4]])))
        elif(labels[i] == 2):
            newLabels = np.vstack((newLabels, np.array([[2,4]])))
        elif(labels[i] == 3):
            newLabels = np.vstack((newLabels, np.array([[0,0]])))
        elif(labels[i] == 4):
            newLabels = np.vstack((newLabels, np.array([[1,0]])))
        elif(labels[i] == 5):
            newLabels = np.vstack((newLabels, np.array([[2,0]])))
        elif(labels[i] == 6):
            newLabels = np.vstack((newLabels,np.array([[0,1]])))
        elif(labels[i] == 7):
            newLabels = np.vstack((newLabels,np.array([[1,1]])))
        elif(labels[i] == 8):
            newLabels = np.vstack((newLabels, np.array([[2,1]])))
        elif(labels[i] == 9):
            newLabels = np.vstack((newLabels, np.array([[0,2]])))
        elif(labels[i] == 10):
            newLabels = np.vstack((newLabels, np.array([[1,2]])))
        else:
            newLabels = np.vstack((newLabels, np.array([[2,2]])))
    newLabels = np.int_(newLabels)
    return newLabels
"""

def toDoubleLabels(labels):
    newLabels = np.empty([0,0])
    newLabels1 = np.empty([0,0])
    for i in range(0,labels.shape[0],1):
        if(labels[i] == 0):
            newLabels = np.append(newLabels, 0)
            newLabels1 = np.append(newLabels1, 3)
        elif(labels[i] == 1):
             newLabels = np.append(newLabels, 1)
             newLabels1 = np.append(newLabels1, 3)
        elif(labels[i] == 2):
             newLabels = np.append(newLabels, 2)
             newLabels1 = np.append(newLabels1, 3)
        elif(labels[i] == 3):
             newLabels = np.append(newLabels, 0)
             newLabels1 = np.append(newLabels1, 0)
        elif(labels[i] == 4):
             newLabels = np.append(newLabels, 1)
             newLabels1 = np.append(newLabels1, 0)
        elif(labels[i] == 5):
             newLabels = np.append(newLabels, 2)
             newLabels1 = np.append(newLabels1, 0)
        elif(labels[i] == 6):
             newLabels = np.append(newLabels, 0)
             newLabels1 = np.append(newLabels1, 1)
        elif(labels[i] == 7):
             newLabels = np.append(newLabels, 1)
             newLabels1 = np.append(newLabels1, 1)
        elif(labels[i] == 8):
             newLabels = np.append(newLabels, 2)
             newLabels1 = np.append(newLabels1, 1)
        elif(labels[i] == 9):
             newLabels = np.append(newLabels, 0)
             newLabels1 = np.append(newLabels1, 2)
        elif(labels[i] == 10):
             newLabels = np.append(newLabels, 1)
             newLabels1 = np.append(newLabels1, 2)
        else:
             newLabels = np.append(newLabels, 2)
             newLabels1 = np.append(newLabels1, 2)
    newLabels = np.int_(newLabels)
    newLabels1 = np.int_(newLabels1)
    return newLabels, newLabels1

"""

file = h5py.File('12LabelsNormalized.h5', 'r')
trainLabels = file['train/labels']
validLabels = file['valid/labels']
testLabels = file['test/labels']

newfile = h5py.File('doubleLabels.h5', 'a')

trainLabels, trainLabels1= toDoubleLabels(trainLabels)
newfile.create_dataset('train/labels/share1', (21060,), dtype='int64', data=trainLabels)
newfile.create_dataset('train/labels/share2', (21060,), dtype='int64', data=trainLabels1)
validLabels, validLabels1= toDoubleLabels(validLabels)
newfile.create_dataset('valid/labels/share1', (7020,), dtype='int64', data=validLabels)
newfile.create_dataset('valid/labels/share2', (7020,), dtype='int64', data=validLabels1)
testLabels, testLabels1= toDoubleLabels(testLabels)
newfile.create_dataset('test/labels/share1', (7020,), dtype='int64', data=testLabels)
newfile.create_dataset('test/labels/share2', (7020,), dtype='int64', data=testLabels1)

newfile.close()
file.close()

"""

######################################################################################
 ### CODICE PER CREARE UN DATASET CON 39 LABELS ###

def to39Labels(labels):
    newLabels = np.empty([0,0])
    for i in range(0,labels.shape[0],1):
        newLabels = np.append(newLabels, labels[i])
    newLabels = np.int_(newLabels)
    return newLabels

"""
# Apro il file h5py
file = h5py.File('dataset.h5', 'r')
trainLabels = file['train/labels']
validLabels = file['valid/labels']
testLabels = file['test/labels']

newfile = h5py.File('39Labels.h5', 'a')

trainLabels= to39Labels(trainLabels)
newfile.create_dataset('train/labels', (21060,), dtype='int64', data=trainLabels)
validLabels= to39Labels(validLabels)
newfile.create_dataset('valid/labels', (7020,), dtype='int64', data=validLabels)
testLabels= to39Labels(testLabels)
newfile.create_dataset('test/labels', (7020,), dtype='int64', data=testLabels)

newfile.close()
file.close()
"""

######################################################################################
 ### CODICE PER CREARE TRE LABELS: UNA SU 3 CLASSI
 ### PER LA PRIMA CONDIVISIONE (3 social) E DUE SU 4 CLASSI PER LA
 ### SECONDA E LA TERZA (3 social + none) ###


def toTripleLabels(labels):
    newLabels = np.empty([0,0])
    newLabels1 = np.empty([0,0])
    newLabels2 = np.empty([0,0])
    for i in range(0,labels.shape[0],1):
        if(labels[i] == 0):
            newLabels = np.append(newLabels, 0)
            newLabels1 = np.append(newLabels1, 3)
            newLabels2 = np.append(newLabels1, 3)
        elif(labels[i] == 1):
             newLabels = np.append(newLabels, 1)
             newLabels1 = np.append(newLabels1, 3)
             newLabels2 = np.append(newLabels1, 3)
        elif(labels[i] == 2):
             newLabels = np.append(newLabels, 2)
             newLabels1 = np.append(newLabels1, 3)
             newLabels2 = np.append(newLabels1, 3)
        elif(labels[i] == 3):
             newLabels = np.append(newLabels, 0)
             newLabels1 = np.append(newLabels1, 0)
             newLabels2 = np.append(newLabels1, 3)
        elif(labels[i] == 4):
             newLabels = np.append(newLabels, 1)
             newLabels1 = np.append(newLabels1, 0)
             newLabels2 = np.append(newLabels1, 3)
        elif(labels[i] == 5):
             newLabels = np.append(newLabels, 2)
             newLabels1 = np.append(newLabels1, 0)
             newLabels2 = np.append(newLabels1, 3)
        elif(labels[i] == 6):
             newLabels = np.append(newLabels, 0)
             newLabels1 = np.append(newLabels1, 1)
             newLabels2 = np.append(newLabels1, 3)
        elif(labels[i] == 7):
             newLabels = np.append(newLabels, 1)
             newLabels1 = np.append(newLabels1, 1)
             newLabels2 = np.append(newLabels1, 3)
        elif(labels[i] == 8):
             newLabels = np.append(newLabels, 2)
             newLabels1 = np.append(newLabels1, 1)
             newLabels2 = np.append(newLabels1, 3)
        elif(labels[i] == 9):
             newLabels = np.append(newLabels, 0)
             newLabels1 = np.append(newLabels1, 2)
             newLabels2 = np.append(newLabels1, 3)
        elif(labels[i] == 10):
             newLabels = np.append(newLabels, 1)
             newLabels1 = np.append(newLabels1, 2)
             newLabels2 = np.append(newLabels1, 3)
        elif(labels[i] == 11):
             newLabels = np.append(newLabels, 2)
             newLabels1 = np.append(newLabels1, 2)
             newLabels2 = np.append(newLabels1, 3)
            # TODO
    newLabels = np.int_(newLabels)
    newLabels1 = np.int_(newLabels1)
    newLabels2 = np.int_(newLabels2)
    return newLabels, newLabels1, newLabels2





