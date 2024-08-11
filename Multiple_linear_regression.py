## import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## import dataset
df=pd.read_csv('50_Startups.csv')
#print(df.head())
#print(df.info())
#print(df.describe())
plt.figure('R&D Spend')
plt.scatter(df['R&D Spend'],df['Profit'])
plt.figure('Administration')
plt.scatter(df['Administration'],df['Profit'])
plt.figure('Marketing Spend')
plt.scatter(df['Marketing Spend'],df['Profit'])
#plt.show()
tehran = df[df['City']=='Tehran']
tabriz = df[df['City']=='Tabriz']
shiraz = df[df['City']=='Shiraz']
#print(tehran.head())
plt.figure('profit')
plt.boxplot([tehran['Profit'], tabriz['Profit'], shiraz['Profit']], labels=['Tehran', 'Tabriz', 'Shiraz'])
#plt.show()
#print(df.head())
newdf=df.drop('City',axis=1)
print(newdf.corr())

## Encoding categorical data
#print(df['City'].unique())
df=pd.get_dummies(df).iloc[:,:-1]
#print(df.head())
##spliting the dataset to traning set and test set
from sklearn.model_selection import train_test_split
X=df.drop('Profit',axis=1).values

Y=df['Profit'].values
#print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

## Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

## Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)

## Predicting the test set results
y_pred=reg.predict(X_test)
#print(y_pred)
#print(y_test)
dff=pd.DataFrame(y_pred)
dff=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_pred)],axis=1)
#print(dff)
print(dff.corr())
## Visualising the Training set resualts


## Visualising the Test set resualts



## Model parameters
print(reg.coef_)
print(reg.intercept_)
##Final Model: Y= 27367 + 8967 * X


#test commit