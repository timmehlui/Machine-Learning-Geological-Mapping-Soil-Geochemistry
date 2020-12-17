# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:55:40 2020

@author: Timothy

Attempt to visualize correlation of data so we can remove highly correlated
data from the data set.

Will test 100 train/test splits and see if the correlation is the same as if using the whole data set.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("/Users/Timothy/Dropbox/Undergrad Thesis Backups/Programming Files Dropbox/dataMasterNoSimpson.csv")
X = data.drop(['Easting', 'Northing', 'GeoUnit'], axis=1)
y = np.array(data['GeoUnit'])
corr_pairs = []
corr = X.corr()
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i, j] >= 0.9:
            corr_pairs.append([i,j])

"""
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(X.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(X.columns)
ax.set_yticklabels(X.columns)
plt.show()
"""
# Create list with 100 0's, to replace with pairs
hundred_pairs = [0]*1000
for k in range(1000):
    seed = k
    corr_pairs_100 = []
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
    corr_100 = X_train.corr()
    for i in range(corr_100.shape[0]):
        for j in range(i+1, corr_100.shape[0]):
            if corr_100.iloc[i, j] >= 0.9:
                corr_pairs_100.append([i,j])
            if corr_100.iloc[i, j] <= -0.9:
                corr_pairs_100.append([i,j])
    hundred_pairs[k] = corr_pairs_100
    
hundred_pairsnp = np.array(hundred_pairs)
unique_hp = np.unique(hundred_pairsnp)

same_indices = [[], [], [], []]
"""
for n in range(len(unique_hp)):
    for m in range(100):
        if (hundred_pairsnp[m] == unique_hp[n]):
            same_indices[n].append(m)
"""
for m in range(1000):
    for n in range(len(unique_hp)):
        if (hundred_pairsnp[m] == unique_hp[n]):
            same_indices[n].append(m)