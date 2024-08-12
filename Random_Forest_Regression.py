## import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## import dataset
df=pd.read_csv('Position_Salaries.csv')
#print(df.head())
#print(df.info())
#print(df.describe())
X=df.iloc[:,1:-1].values
Y=df.iloc[:,-1].values

## Training the Random forest model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
fr=RandomForestRegressor(n_estimators=20)
fr.fit(X,Y)

## Visualising the Random Forest Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
print(X_grid)
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, fr.predict(X_grid), color = 'blue')
plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

## Predicting a new result with  Random Forest Regression
print(fr.predict([[5.5]]))

