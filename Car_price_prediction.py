## importing libaries and dadtaset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

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
print(df.isna().sum())
df.info()


