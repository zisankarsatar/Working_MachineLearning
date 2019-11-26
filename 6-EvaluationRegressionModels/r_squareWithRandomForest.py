# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 03:57:45 2019

@author: ZİŞAN
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("random_forest_regression_dataset.csv", sep=";", header=None)
x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)
#%%
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x,y)

y_head=rf.predict(x)
#%% 
from sklearn.metrics import r2_score

print("r_score:", r2_score(y,y_head))