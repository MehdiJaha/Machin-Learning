## import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## import dataset
df=pd.read_csv('Position_Salaries.csv')
#print(df.info())
#print(df.describe())
X=df.iloc[:,1:-1].values
Y=df.iloc[:,-1].values
#plt.scatter(X,Y,color='red')
#plt.show()
## Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lreg=LinearRegression()
lreg.fit(X,Y)

#plt.scatter(X, Y)
#plt.plot(X, lreg.predict(X), color='blue')
#plt.show()

## Training the polynomial regression on whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures()

X_poly=poly.fit_transform(X)
reg=LinearRegression()
reg.fit(X_poly,Y)

## Visualising the Polynomial Regression results
#plt.scatter(X, Y)
#plt.plot(X, reg.predict(X_poly), color='red')
#plt.show()

## Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y)
plt.plot(X_grid, reg.predict(poly.fit_transform(X_grid)), color = 'red')
plt.show()

## Predicting a new result with Linear Regression
lreg.predict([[6.5]])

## Predicting a new result with Polynomial Regression
reg.predict(poly.fit_transform([[6.5]]))

