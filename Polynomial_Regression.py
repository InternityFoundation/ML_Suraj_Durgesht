# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:58:51 2019

@author: Suraj
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

######################################

datas = pd.read_csv('data.csv')
print(datas)

###############################

X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values

################################

from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X,y)

 ##################################
 
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)

poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)
#######################

plt.scatter(X, y,color = 'green')
plt.plot(X, lin.predict(X), color = 'red')
plt.title('Linear regression')
plt.xlabel('Temprature')
plt.ylabel('Pressure')
plt.show()
##################################

plt.scatter(X, y,color = 'blue')
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red')
plt.title('Linear regression')
plt.xlabel('Temprature')
plt.ylabel('Pressure')
plt.show()
#####################################
