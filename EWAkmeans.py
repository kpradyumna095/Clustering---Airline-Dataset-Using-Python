# -*- coding: utf-8 -*-
"""
Created on Fri May 31 22:20:41 2019

@author: GANGADHAR
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

flight=pd.read_excel('C:\\Users\\GANGADHAR\\Desktop\\sainath assignments\\Rcodes\\clustering\\flight.xlsx')

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)
    
df_norm=norm_func(flight.iloc[:,1:])

k=list(range(10,20))
k
TWSS=[]
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
TWSS

plt.plot(k,TWSS, 'ro-');plt.xlabel('number of clusters');plt.ylabel('total within sum of squares');plt.xticks(k)
    
model1=KMeans(n_clusters=14)
model1.fit(df_norm)

model1.cluster_centers_
model1.labels_
model=pd.Series(model1.labels_)
model
flight['clust']=model



flightfinal=flight.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]

fly=flight.iloc[:,1:13].groupby(flightfinal.clust).mean()

flightfinal.to_csv("flightfinalkmewa.csv",encoding="utf-8")
