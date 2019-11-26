# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:23:39 2019

@author: ZİŞAN
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv("data(11).csv")
data.drop(["id","Unnamed: 32"], axis=1, inplace=True)
data.head()

#%%
#malignant = M kötü huylu tümör
#benign =B iyi huylu tümör
M=data[data.diagnosis=="M"]
B=data[data.diagnosis=="B"]

#M.info()
#B.info()
#%% aplha parametresi saydamlık verşyor grafiğe 
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu", alpha=0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha=0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()
#%%
data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)
#%% normalizasyon
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#%%
from sklearn.model_selection import train_test_split
x_train, y_train, x_test, y_test =train_test_split(x,y,test_size=0.3,random_state=1)
#%%
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train, y_train)
#%%test
print("print accuracy of svm algo: ",nb.score(x_test,y_test))