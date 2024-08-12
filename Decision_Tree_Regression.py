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

## Training the Decision Tree model on the whole dataset

from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor()
reg.fit(X,Y)

## Visualising the Decision Tree Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
print(X_grid)
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

## Predicting a new result with Decision Tree Regression
print(reg.predict([[5.5]]))

