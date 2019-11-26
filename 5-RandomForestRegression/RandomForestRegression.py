# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 03:27:57 2019

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

print("7.8 seviyesindeki trübünün ne kadar olduğu:", rf.predict([[7.8]]))
x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=rf.predict(x_)
#%% visualize

plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("tribün level")
plt.ylabel("ücret")
plt.show()