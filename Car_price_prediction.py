## importing libaries and dadtaset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

data=pd.read_csv('car details.csv')
#print(data.head())

df=data.copy()
print(df.describe(include='all'))



