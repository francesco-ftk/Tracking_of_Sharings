import matplotlib.pyplot as plt
import numpy as np
import h5py

f = h5py.File('ProbabilityDistribution.h5', 'r')

"""
x= f['share1/true']
y= np.empty([0,0])
for i in range(len(x)):
    m = max(x[i])
    y = np.concatenate((y,m), axis=None)

fig = plt.figure(figsize =(6, 8))
plt.boxplot(y)
plt.title("share1 true")
plt.show()

x= f['share2/true']
y= np.empty([0,0])
for i in range(len(x)):
    m = max(x[i])
    y = np.concatenate((y,m), axis=None)

fig = plt.figure(figsize =(6, 8))
plt.boxplot(y)
plt.title("share2 true")
plt.show()

x= f['share2/false']
y= np.empty([0,0])
for i in range(len(x)):
    m = max(x[i])
    y = np.concatenate((y,m), axis=None)

fig = plt.figure(figsize =(6, 8))
plt.boxplot(y)
plt.title("share2 false")
plt.show()

x= f['share3/true']
y= np.empty([0,0])
for i in range(len(x)):
    m = max(x[i])
    y = np.concatenate((y,m), axis=None)

fig = plt.figure(figsize =(6, 8))
plt.boxplot(y)
plt.title("share3 true")
plt.show()

x= f['share3/false']
y= np.empty([0,0])
for i in range(len(x)):
    m = max(x[i])
    y = np.concatenate((y,m), axis=None)

fig = plt.figure(figsize =(6, 8))
plt.boxplot(y)
plt.title("share3 false")
plt.show()

"""

t1 = np.empty([0,0])
t2 = np.empty([0,0])
f2 = np.empty([0,0])
t3 = np.empty([0,0])
f3 = np.empty([0,0])

x1= f['share1/true']
for i in range(len(x1)):
    m = max(x1[i])
    t1 = np.concatenate((t1,m), axis=None)

x2= f['share2/true']
for i in range(len(x2)):
    m = max(x2[i])
    t2 = np.concatenate((t2,m), axis=None)

x3= f['share2/false']
for i in range(len(x3)):
    m = max(x3[i])
    f2 = np.concatenate((f2,m), axis=None)

x4= f['share3/true']
for i in range(len(x4)):
    m = max(x4[i])
    t3 = np.concatenate((t3,m), axis=None)

x5= f['share3/false']
for i in range(len(x5)):
    m = max(x5[i])
    f3 = np.concatenate((f3,m), axis=None)

labels = ['true share1', 'true share2', 'false share2', 'true share3', 'false share3']
fig = plt.figure(figsize =(12, 8))
plt.boxplot([t1,t2,f2,t3,f3], labels=labels)
plt.show()
