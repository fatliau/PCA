#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:29:59 2017

@author: JC
"""
import numpy as np

#%% PCA1
Xt = np.genfromtxt('wine.data', delimiter=',', autostrip=True)
#X = np.matrix('1 2; 3 4')
#y = np.genfromtxt('wine.labels', delimiter=',', autostrip=True)

print("len(Xt)=",len(Xt))
print("len(Xt[0])=",len(Xt[0]))

evals, evecs = np.linalg.eig(np.dot(Xt.T,Xt)); print("evals=", evals); print("evecs=", evecs)
#evals, evecs = np.linalg.eig(X); print("evals=", evals); print("evecs=", evecs)

print("len(evals)=",len(evals))

idx = np.argsort(evals); print("idx=",idx); print("len(idx)=",len(idx))
print("evals=",evals)

idx1 = np.argsort(evals)[-1]; print("idx1=",idx1)
idx2 = np.argsort(evals)[-2]; print("idx2=",idx2)

newevecs=[]
for i in range(len(evecs)):
    #if i % 15 ==0:
    #    print(newevecs[i])
    newevecs.append([evecs[i,idx1],evecs[i,idx2]])
    #newevecs.append([evecs[i,idx1]])
    
print("newevecs=",newevecs)
print("len(newevecs)=",len(newevecs))

print("np.array(newevecs)=",np.array(newevecs))
print("len(np.array(newevecs))=",len(np.array(newevecs)))
print("len(np.array(newevecs[0]))=",len(np.array(newevecs[0])))

projection = np.dot(Xt,np.array(newevecs)); print("projection=",projection)
print("len(projection)=",len(projection))
print("len(projection[0])=",len(projection[0]))

#%% PCA2
Xt = np.genfromtxt('wine.data', delimiter=',', autostrip=True)

print("len(Xt)=",len(Xt))

mean_all = np.mean(Xt);print("mean_all=",mean_all)
evals, evecs = np.linalg.eig(np.dot(Xt.T,Xt)-mean_all); print("evals=", evals); print("evecs=", evecs)

print("len(evals)=",len(evals))

idx = np.argsort(evals); print("idx=",idx); print("len(idx)=",len(idx))
print("evals=",evals)

idx1 = np.argsort(evals)[-1]; print("idx1=",idx1)
idx2 = np.argsort(evals)[-2]; print("idx2=",idx2)

newevecs=[]
for i in range(len(evecs)):
    #if i % 15 ==0:
    #    print(newevecs[i])
    newevecs.append([evecs[i,idx1],evecs[i,idx2]])
    #newevecs.append([evecs[i,idx1]])
    
print("newevecs=",newevecs)
print("len(newevecs)=",len(newevecs))

print("np.array(newevecs)=",np.array(newevecs))
print("len(np.array(newevecs))=",len(np.array(newevecs)))
print("len(np.array(newevecs[0]))=",len(np.array(newevecs[0])))

projection = np.dot(Xt,np.array(newevecs)); print("projection=",projection)
print("len(projection)=",len(projection))
print("len(projection[0])=",len(projection[0]))

#%% within scatter

Xt = np.genfromtxt('wine.data', delimiter=',', autostrip=True)
y = np.genfromtxt('wine.labels', delimiter=',', autostrip=True)
#Xt = np.matrix('1 2 1 2; 3 4 1 2; 1 2 1 3')
#y = np.matrix('1; 2; 1')
#Xt = Xt[:4]
#y = y[:4]

n,m = Xt.shape; print("n=",n,";m=",m)
#for i in range(5):
#    print(Xt[i],y[i])

#datalen = len(y); print("datalen=",datalen)

#to see how many possible lable in y
labels=[]
for i in range(len(y)):
    if y[i] not in labels:
        labels.append(y[i])
        
print("labels=",labels)

sumation = np.zeros((len(labels),m)); print("sumation=",sumation)
counts = np.zeros((len(labels)))
#mean_label = []

for i, xt in enumerate(Xt):
    for l in range(len(labels)):
        if y[i] == labels[l]:
            counts[l] += 1
            sumation[l] += xt

print("sumation=",sumation); print("counts=",counts)

mean = np.zeros((len(labels),m));
for i, xt in enumerate(sumation):
    mean[i] = xt/counts[i]
print("mean=",mean)
print("mean.shape=",mean.shape)

#print(labels[0],labels[1])      


C = np.zeros((m,m))
"""
for i, xt in enumerate(Xt):
    ct = xt - mean; print("ct=",ct)
    temp = np.dot(ct.T,ct); print("np.dot(ct.T,ct)=",temp)
    C += temp
"""
for i, xt in enumerate(Xt):
    for l in range(len(labels)):
        if y[i] == labels[l]:
            ct = xt - mean[l]; #print("ct=",ct)
            temp = np.outer(ct,ct); #print("np.outer(ct,ct)=",temp)
            C += temp

#print("Xt=",Xt)

   
print("C=",C)
print("C.shpae=",C.shape)

evals, evecs = np.linalg.eig(C); print("evals=", evals); print("evecs=", evecs)

print("len(evals)=",len(evals))

idx = np.argsort(evals); print("idx=",idx); print("len(idx)=",len(idx))
print("evals=",evals)

idx1 = np.argsort(evals)[0]; print("idx1=",idx1)
idx2 = np.argsort(evals)[1]; print("idx2=",idx2)

newevecs=[]
for i in range(len(evecs)):
    #if i % 15 ==0:
    #    print(newevecs[i])
    newevecs.append([evecs[i,idx1],evecs[i,idx2]])
    #newevecs.append([evecs[i,idx1]])
    
print("newevecs=",newevecs)
print("len(newevecs)=",len(newevecs))

print("np.array(newevecs)=",np.array(newevecs))
print("len(np.array(newevecs))=",len(np.array(newevecs)))
print("len(np.array(newevecs[0]))=",len(np.array(newevecs[0])))

projection = np.dot(Xt,np.array(newevecs)); print("projection=",projection)
print("len(projection)=",len(projection))
print("len(projection[0])=",len(projection[0]))

#%% between scatter

#Xt = np.genfromtxt('testdata.txt', delimiter=',', autostrip=True)
#y = np.genfromtxt('testlabel.txt', delimiter=',', autostrip=True)
Xt = np.genfromtxt('wine.data', delimiter=',', autostrip=True)
y = np.genfromtxt('wine.labels', delimiter=',', autostrip=True)
#Xt = np.matrix('1 2 1 2; 3 4 1 2; 1 2 1 3')
#y = np.matrix('1; 2; 1')
#Xt = Xt[:4]
#y = y[:4]
#print(Xt)
#print(y)

n,m = Xt.shape; print("n=",n,";m=",m)
#for i in range(5):
#    print(Xt[i],y[i])

#datalen = len(y); print("datalen=",datalen)

#to see how many possible lable in y
labels=[]
for i in range(len(y)):
    if y[i] not in labels:
        labels.append(y[i])
        
print("labels=",labels)

sumation = np.zeros((len(labels),m)); print("sumation=",sumation)
counts = np.zeros((len(labels)))
#mean_label = []
sumall = np.zeros(m)
countall = 0
for i, xt in enumerate(Xt):
    for l in range(len(labels)):
        if y[i] == labels[l]:
            counts[l] += 1
            sumation[l] += xt
            sumall += xt
            countall += 1

print("sumall=",sumall)
print("countall=",countall)

mean_all = sumall/countall
print("mean_all=",mean_all)
print("mean_all.shape=",mean_all.shape)

print("sumation=",sumation); print("counts=",counts)

mean = np.zeros((len(labels),m));
for i, xt in enumerate(sumation):
    mean[i] = xt/counts[i]
print("mean=",mean)
print("mean.shape=",mean.shape)

C = np.zeros((m,m))
for i, xt in enumerate(mean):
    ct = xt - mean_all
    C += counts[i] * np.outer(ct,ct)

print("C=",C)
print("C.shapre=",C.shape)


evals, evecs = np.linalg.eig(C); print("evals=", evals); print("evecs=", evecs)

print("len(evals)=",len(evals))

idx = np.argsort(evals); print("idx=",idx); print("len(idx)=",len(idx))
print("evals=",evals)

idx1 = np.argsort(evals)[-1]; print("idx1=",idx1)
idx2 = np.argsort(evals)[-2]; print("idx2=",idx2)

newevecs=[]
for i in range(len(evecs)):
    #if i % 15 ==0:
    #    print(newevecs[i])
    newevecs.append([evecs[i,idx1],evecs[i,idx2]])
    #newevecs.append([evecs[i,idx1]])
    
print("newevecs=",newevecs)
print("len(newevecs)=",len(newevecs))

print("np.array(newevecs)=",np.array(newevecs))
print("len(np.array(newevecs))=",len(np.array(newevecs)))
print("len(np.array(newevecs[0]))=",len(np.array(newevecs[0])))

projection = np.dot(Xt,np.array(newevecs)); print("projection=",projection)
print("len(projection)=",len(projection))
print("len(projection[0])=",len(projection[0]))

# %% ratio of between and within

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

#Xt = np.genfromtxt('testdata.txt', delimiter=',', autostrip=True)
#y = np.genfromtxt('testlabel.txt', delimiter=',', autostrip=True)
Xt = np.genfromtxt('wine.data', delimiter=',', autostrip=True)
y = np.genfromtxt('wine.labels', delimiter=',', autostrip=True)

n,m = Xt.shape; print("n=",n,";m=",m)

#to see how many possible lable in y
labels=[]
for i in range(len(y)):
    if y[i] not in labels:
        labels.append(y[i])
        
print("labels=",labels)

sumation = np.zeros((len(labels),m)); print("sumation=",sumation)
counts = np.zeros((len(labels)))
#mean_label = []
sumall = np.zeros(m)
countall = 0
for i, xt in enumerate(Xt):
    for l in range(len(labels)):
        if y[i] == labels[l]:
            counts[l] += 1
            sumation[l] += xt
            sumall += xt
            countall += 1

print("sumall=",sumall)
print("countall=",countall)

mean_all = sumall/countall
print("mean_all=",mean_all)
print("mean_all.shape=",mean_all.shape)

print("sumation=",sumation); print("counts=",counts)

mean = np.zeros((len(labels),m));
for i, xt in enumerate(sumation):
    mean[i] = xt/counts[i]
print("mean=",mean)
print("mean.shape=",mean.shape)


W = np.zeros((m,m))
for i, xt in enumerate(Xt):
    for l in range(len(labels)):
        if y[i] == labels[l]:
            ct = xt - mean[l]; #print("ct=",ct)
            temp = np.outer(ct,ct); #print("np.outer(ct,ct)=",temp)
            W += temp

#print("Xt=",Xt)
   
print("W=",W)
print("W.shpae=",W.shape)

Winv = np.linalg.inv(W); print("Winv=",Winv) 
#print("W*Winv=",np.dot(W,Winv))

B = np.zeros((m,m))
for i, xt in enumerate(mean):
    ct = xt - mean_all
    B += counts[i] * np.outer(ct,ct)

print("B=",B)
print("B.shapre=",B.shape)

WinvB = np.dot(Winv,B); print("WinvB=",WinvB)

evals, evecs = np.linalg.eig(WinvB); print("evals=", evals); print("evecs=", evecs)

print("len(evals)=",len(evals))

idx = np.argsort(evals); print("idx=",idx); print("len(idx)=",len(idx))
print("evals=",evals)

idx1 = np.argsort(evals)[-1]; print("idx1=",idx1)
idx2 = np.argsort(evals)[-2]; print("idx2=",idx2)

newevecs=[]
for i in range(len(evecs)):
    #if i % 15 ==0:
    #    print(newevecs[i])
    newevecs.append([evecs[i,idx1],evecs[i,idx2]])
    #newevecs.append([evecs[i,idx1]])
    
print("newevecs=",newevecs)
print("len(newevecs)=",len(newevecs))

print("np.array(newevecs)=",np.array(newevecs))
print("len(np.array(newevecs))=",len(np.array(newevecs)))
print("len(np.array(newevecs[0]))=",len(np.array(newevecs[0])))

projection = np.dot(Xt,np.array(newevecs)); print("projection=",projection)
print("len(projection)=",len(projection))
print("len(projection[0])=",len(projection[0]))

# %% test WinvB

#float_formatter = lambda x: "%.4f" % x
#np.set_printoptions(formatter={'float_kind':float_formatter})

Xt = np.genfromtxt('testdata.txt', delimiter=',', autostrip=True)
y = np.genfromtxt('testlabel.txt', delimiter=',', autostrip=True)
#Xt = np.genfromtxt('wine.data', delimiter=',', autostrip=True)
#y = np.genfromtxt('wine.labels', delimiter=',', autostrip=True)

n,m = Xt.shape; print("n=",n,";m=",m)

#to see how many possible lable in y
labels=[]
for i in range(len(y)):
    if y[i] not in labels:
        labels.append(y[i])
        
print("labels=",labels)

sumation = np.zeros((len(labels),m)); print("sumation=",sumation)
counts = np.zeros((len(labels)))
#mean_label = []
sumall = np.zeros(m)
countall = 0
for i, xt in enumerate(Xt):
    for l in range(len(labels)):
        if y[i] == labels[l]:
            counts[l] += 1
            sumation[l] += xt
            sumall += xt
            countall += 1

print("sumall=",sumall)
print("countall=",countall)

mean_all = sumall/countall
print("mean_all=",mean_all)
print("mean_all.shape=",mean_all.shape)

print("sumation=",sumation); print("counts=",counts)

mean = np.zeros((len(labels),m));
for i, xt in enumerate(sumation):
    mean[i] = xt/counts[i]
print("mean=",mean)
print("mean.shape=",mean.shape)


W = np.zeros((m,m))
for i, xt in enumerate(Xt):
    for l in range(len(labels)):
        if y[i] == labels[l]:
            ct = xt - mean[l]; #print("ct=",ct)
            temp = np.outer(ct,ct); #print("np.outer(ct,ct)=",temp)
            W += temp

#print("Xt=",Xt)
   
print("W=",W)
print("W.shpae=",W.shape)

Winv = np.linalg.inv(W); print("Winv=",Winv) 
#print("W*Winv=",np.dot(W,Winv))

B = np.zeros((m,m))
for i, xt in enumerate(mean):
    ct = xt - mean_all
    B += counts[i] * np.outer(ct,ct)

print("B=",B)
print("B.shapre=",B.shape)

WinvB = np.dot(Winv,B); print("WinvB=",WinvB)

evals, evecs = np.linalg.eig(WinvB); print("evals=", evals); print("evecs=", evecs)

print("len(evals)=",len(evals))

idx = np.argsort(evals); print("idx=",idx); print("len(idx)=",len(idx))
print("evals=",evals)

idx1 = np.argsort(evals)[1]; print("idx1=",idx1)

newevecs=[]
for i in range(len(evecs)):
    #if i % 15 ==0:
    #    print(newevecs[i])
    newevecs.append(evecs[i,idx1])
    #newevecs.append([evecs[i,idx1]])
    
print("newevecs=",newevecs)
print("len(newevecs)=",len(newevecs))

print("np.array(newevecs)=",np.array(newevecs))
print("len(np.array(newevecs))=",len(np.array(newevecs)))
#print("len(np.array(newevecs[0]))=",len(np.array(newevecs[0])))

projection = np.dot(Xt,np.array(newevecs)); print("projection=",projection)
print("len(projection)=",len(projection))
#print("len(projection[0])=",len(projection[0]))
