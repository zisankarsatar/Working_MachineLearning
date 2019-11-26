# -*- coding: utf-8 -*

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("polynomial_regression.csv", sep=";")

x=df.araba_fiyat.values.reshape(-1,1)
y=df.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("Araba_Fiyat")
plt.ylabel("Araba_Hiz")
plt.show()
#%% polynomial regresson = y=b0+b1x+b2x2+...bnxn

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression=PolynomialFeatures(degree=2)

x_polynomial=polynomial_regression.fit_transform(x)
#%%fit
from sklearn.linear_model import LinearRegression
linear_regression=LinearRegression()
linear_regression.fit(x_polynomial,y)
#%%

y_head=linear_regression.predict(x_polynomial)

plt.plot(x,y_head,color="green", label="poly")
plt.legend()
plt.show()

