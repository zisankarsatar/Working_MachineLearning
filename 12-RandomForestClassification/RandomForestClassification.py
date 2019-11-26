import pandas as pd
import numpy as np

#%%gereksiz veriler atıldı
data=pd.read_csv("data (14).csv")
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
#%%stringler sayısala dönüşt ürüldü
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

print("decision tree score:", dt.score(x_test,y_test))
#%%random forest algoritması
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(x_train,y_train)
print("random forest algo score:", rf.score(x_test,y_test))