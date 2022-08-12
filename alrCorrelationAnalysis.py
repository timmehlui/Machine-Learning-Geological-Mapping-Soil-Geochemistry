# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:11:58 2022

@author: Timmy

ALR Data Analysis to understand what features to remove based on correlation
analysis. Will divide by Ti since it's immobile. 

"""

# Import stuff
import pandas as pd
import pyrolite.comp
import numpy as np
import matplotlib.pyplot as plt

# Import data
# Doing it on no test data so that doesn't affect our analysis
data_og = pd.read_csv("dataMasterFiveNoTest.csv")
data_gc = data_og.drop(['Easting', 'Northing', 'NGU_orig', 'OGU', 'InNewSet', 'NGU', 
                        'SlopeRep', 'AspectRep', 'ElevRep'], axis = 1)


alrdata = data_gc.pyrocomp.ALR(ind = 'Ti')

corr = alrdata.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(alrdata.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(alrdata.columns)
ax.set_yticklabels(alrdata.columns)
plt.show()

highcorr = corr[corr>0.8]
# Ca: Sr
# Co: Fe
# Cr: Ni
# Fe: V
# ^0.9 v0.8


"""
# Try a different column
alrdata = data_gc.pyrocomp.ALR(ind = 'Sc')

corr = alrdata.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(alrdata.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(alrdata.columns)
ax.set_yticklabels(alrdata.columns)
plt.show()

highcorr = corr[corr>0.8]
# Ni: Cr
# ^0.9 v0.8
# Al : Cr
# Ca: Sr
# Cd: Zn
# Co: Fe
# Cr: Al, Mg, Ni
# Fe: Co, V
# Pb: Zn
"""
