## import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## import dataset
df=pd.read_csv('Position_Salaries.csv')
#print(df.head())
#print(df.info())
#print(df.describe())s
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values.reshape(-1,1)


## Feature Scaling
from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(x)
Y = sc_y.fit_transform(y)


## Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X, Y.ravel())

## Predicting a new result
print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[5.5]])).reshape(1,-1)))
