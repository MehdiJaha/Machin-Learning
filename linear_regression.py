
## import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## import dataset
df=pd.read_csv('Salary_Data.csv')
#print(df.head())
#print(df.info())
#print(df.describe())

X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values

#print(Y)

##spliting the dataset to traning set and test set

from sklearn.model_selection import train_test_split

X_train ,X_test ,Y_train ,Y_test =train_test_split(X,Y, test_size=0.3)

from sklearn.linear_model import LinearRegression

## Training the Simple Linear Regression model on the Training set
reg=LinearRegression()
reg.fit(X_train,Y_train)

## Predicting the test set results

Y_pred=reg.predict(X_test)
#print(Y_pred)

## Visualising the Training set resualts
plt.figure('Training set resualts')
plt.scatter(X_train,Y_train)
plt.plot(X_test,reg.predict(X_test),color='red')

## Visualising the Test set resualts
plt.figure('Test set resualts')
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, reg.predict(X_test), color='blue')


## Model parameters

print(reg.coef_)
print(reg.intercept_)
##Final Model: Y= 27367 + 8967 * X

plt.show()



