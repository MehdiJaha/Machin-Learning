## import libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , accuracy_score
df=pd.read_csv('Social_Network_Ads.csv')
#print (df.info())
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
## spliting the dataset to training set and test set

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

## feature scalling

sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

##trainning the K_NN model on the trainning set
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5 , )
classifier.fit(X_train,Y_train)

## Predict new resualt 
print(classifier.predict(sc.transform([[30,90000]])))

##predict the test set resualt

pred=classifier.predict(X_test)
print(pd.concat([pd.DataFrame(pred),pd.DataFrame(Y_test)],axis=1))

##making the confusion matrix
cm=confusion_matrix(pred , Y_test)
print(cm)
print (accuracy_score(pred ,Y_test))

## Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()