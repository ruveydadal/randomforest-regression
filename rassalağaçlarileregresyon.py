# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 18:16:16 2021

@author: muham
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler=pd.read_csv('maaslar.csv')

#verilerin bölünmesi
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
#dataframeleri numpy arraylere dönüştürme
X = x.values
Y = y.values

#linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y)
plt.plot(x,lin_reg.predict(x))

#polynomial regression(2.dereceden)
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x)
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.show()

#polynomial regression(4.dereceden)
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.show()

#random forest regression
#n_estimators -> karar ağacı sayısı
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')







