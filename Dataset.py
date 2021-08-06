import h5py
import numpy as np

# TestSet e ValidationSet di 7020 immagini, 39 etichette possibili, per ognuna abbiamo 369 dct, 10 header e 152 meta, ossia 532 features
# TrainSet di 21060 immagini, 39 etichette possibili, per ognuna abbiamo 369 dct, 10 header e 152 meta, ossia 532 features

######################################################################################
 ### CODICE PER CREARE UN DATASET CON FEATURES NORMALIZZATE E 3 CLASSI ###

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

newfile = h5py.File('3LabelsNormalized.h5', 'a')

Normalize(file, newfile)

trainLabels= reduceTo3Labels(trainLabels)
newfile.create_dataset('train/labels', (21060,), dtype='int64', data=trainLabels)
validLabels= reduceTo3Labels(validLabels)
newfile.create_dataset('valid/labels', (7020,), dtype='int64', data=validLabels)
testLabels= reduceTo3Labels(testLabels)
newfile.create_dataset('test/labels', (7020,), dtype='int64', data=testLabels)

newfile.close()
file.close()
