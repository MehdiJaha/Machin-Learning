## importing libaries and dadtaset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import metrics
from datetime import date
def change_unit(x):
    return 9.8*x if x <= 50 else x

data=pd.read_csv('car details.csv')
#print(data.head())

df=data.copy()
#print(df.describe(include='all'))

## Data preprocessing
##filling missing value


#print(df.isna().sum())
df[['mileage_number','mileage_unit']]=df['mileage'].str.split(' ',n=1,expand=True)
#print(df.head())
#print(df['mileage_unit'].unique())
df.drop(['mileage','mileage_unit'],axis=1,inplace=True)
#print(df.head())


df[['engine_number','engine_unit']]=df['engine'].str.split(' ',n=1,expand=True)
#print(df.head())
#print(df['engine_unit'].unique())
df.drop(['engine','engine_unit'],axis=1,inplace=True)
#print(df['engine_number'].head())


df[['max_power_number','max_power_unit']]=df['max_power'].str.split(' ',n=1,expand=True)
#print(df.head())
#print(df['max_power_unit'].unique())
df.drop(['max_power','max_power_unit'],axis=1,inplace=True)
#print(df['max_power_number'].head())


df[['torque_number', 'torque_unit']] = df['torque'].str.split('@', n=1, expand=True)
#print(df.head())
df['torque_number'] = df['torque_number'].str.extract('(^\d*)')
df.drop(['torque_unit', 'torque'], axis=1, inplace=True)
#df.head()
df['torque_number'] = df['torque_number'].astype(float)


df['torque_number'] = df['torque_number'].apply(change_unit)
#print(df.head())

#print(df.isna().sum())
df['seats'].fillna(df['seats'].mean(),inplace=True)

df['mileage_number'] = df['mileage_number'].astype(float)
df['mileage_number'].fillna(df['mileage_number'].mean(), inplace=True)

df['engine_number'] = df['engine_number'].astype(float)
df['engine_number'].fillna(df['engine_number'].mean(), inplace=True)

df['max_power_number'].replace({'': np.nan}, inplace=True)
df['max_power_number'] = df['max_power_number'].astype(float)
df['max_power_number'].fillna(df['max_power_number'].mean(), inplace=True)

df['torque_number'] = df['torque_number'].astype(float)
df['torque_number'].fillna(df['torque_number'].mean(), inplace=True)
#print(df.isna().sum())
#df.info()

## Encoding categorical features
df.drop(['name'],axis=1,inplace=True)

categorical_cols = []
for col in df.columns:
    if df[col].dtypes == 'object':
        categorical_cols.append(col)
#print(categorical_cols)

df=pd.concat([df, pd.get_dummies(df[categorical_cols],drop_first=True)] , axis=1)

df.drop(categorical_cols,axis=1,inplace=True)

for col in df.columns:
    if df[col].dtypes == 'bool':
        df[col]=df[col].replace({True: 1, False: 0})

#print(df.info())

#print (date.today().year)
df['car_age']=abs(df['year']-date.today().year)
df.drop(['year'],axis=1,inplace=True)
#print(df)

## MOdel building
## Train test split and scaling
X=df.iloc[:,1:]
Y=df.iloc[:,0]

X_train ,X_test ,Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
##linear regression
regressor_lr=LinearRegression()
regressor_lr.fit(X_train,Y_train)

print(regressor_lr.score(X_train,Y_train))
print(regressor_lr.score(X_test,Y_test))

pred_lr=regressor_lr.predict(X_test)
#plt.hist(pred_lr-Y_test,bins=20)
#plt.show()
#plt.scatter(pred_lr,Y_test)
#plt.show()


print('MAE = ', metrics.mean_absolute_error(Y_test, pred_lr))
print('MSE = ', metrics.mean_squared_error(Y_test, pred_lr))
print('RMSE = ', np.sqrt(metrics.mean_squared_error(Y_test, pred_lr)))
print('R2 = ', metrics.r2_score(Y_test, pred_lr))

##Random forest regression
regressor_fr=RandomForestRegressor()
regressor_fr.fit(X_train,Y_train)

print(regressor_fr.score(X_train,Y_train))
print(regressor_fr.score(X_test,Y_test))

pred_fr=regressor_fr.predict(X_test)
#plt.hist(pred_fr-Y_test,bins=20)
#plt.show()
#plt.scatter(pred_fr,Y_test)
#plt.show()

print('MAE = ', metrics.mean_absolute_error(Y_test, pred_fr))
print('MSE = ', metrics.mean_squared_error(Y_test, pred_fr))
print('RMSE = ', np.sqrt(metrics.mean_squared_error(Y_test, pred_fr)))
print('R2 = ', metrics.r2_score(Y_test, pred_fr))

##SVR

sc2=StandardScaler()
Y_train=sc2.fit_transform(pd.DataFrame(Y_train))
Y_test=sc2.transform(pd.DataFrame(Y_test))

regressor_sv=SVR()
regressor_sv.fit(X_train,Y_train.ravel())

print(regressor_sv.score(X_train,Y_train))
print(regressor_sv.score(X_test,Y_test))

pred_sv=regressor_sv.predict(X_test)
#plt.hist(pred_sv.reshape(-1,1)-Y_test,bins=20)
#plt.show()
#plt.scatter(pred_sv,Y_test)
#plt.show()

print('MAE = ', metrics.mean_absolute_error(sc2.inverse_transform(Y_test.reshape(-1, 1)), sc2.inverse_transform(pred_sv.reshape(-1, 1))))
print('MSE = ', metrics.mean_squared_error(sc2.inverse_transform(Y_test.reshape(-1, 1)), sc2.inverse_transform(pred_sv.reshape(-1, 1))))
print('RMSE = ', np.sqrt(metrics.mean_squared_error(sc2.inverse_transform(Y_test.reshape(-1, 1)), sc2.inverse_transform(pred_sv.reshape(-1, 1)))))
print('R2 = ', metrics.r2_score(sc2.inverse_transform(Y_test.reshape(-1, 1)), sc2.inverse_transform(pred_sv.reshape(-1, 1))))