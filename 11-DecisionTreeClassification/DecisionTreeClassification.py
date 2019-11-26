# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:01:49 2019

@author: ZİŞAN
"""

import pandas as pd
import numpy as np
#%%gereksiz veriler atıldı
data=pd.read_csv("data(13).csv")
data.drop(["id","Unnamed:32"],axis=1,inplace=True)
#%%stringler sayısala dönüştürüldü
data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
x_data= data.drop(["diagnosis"],axis=1)
#%%normalizasyon
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#%% decision tree algoritmayı kulanrak dataset train edilcek ve ardından predict edilcek

from sklearn.model_selection import train_test_split
x_train,  x_test,y_train,y_test =train_test_split(x,y,test_size=0.15,random_state= 42)
#%%test
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train, y_train)

print("score:", dt.score(x_test,y_test))