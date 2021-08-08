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

