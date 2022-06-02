from google.colab import files
uploaded = files.upload()

#Execute the above command to Upload Dataset


#Execute the below commands to import missing libraries
!pip install git+https://github.com/jundongl/scikit-feature.git
pip install xlsxwriter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns



#Read data in X and Y # load the dataset
df = pd.read_csv('D1.csv')
# split into input and output variables
x = df.iloc[:, :-1]
y = df.iloc[:,-1]
print(x)
print(y)


#Data preprocessing
x=x.loc[:,(df!=0).any(axis=0)];
scaler=MinMaxScaler(copy=True,feature_range=(0,1));
x_scaled=scaler.fit_transform(x)
x=x_scaled


# use train and test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)


#KNN CLASSIFIER accuracy
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))


# 10-fold cross-validation for KNN
f_measure_knn = cross_val_score(knn,x,y,cv=10,scoring='f1_weighted')
#accuracy_knn  = cross_val_score(knn,x_train,y_train,cv=10, scoring='accuracy')
accuracy_knn  = cross_val_score(knn,x,y,cv=10, scoring='accuracy')
accuracy_knn=accuracy_knn*100
print(f_measure_knn)
print(accuracy_knn)

#RANDOM FOREST CLASSIFIER 
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))

# 10-fold cross-validation for RF
f_measure_rf = cross_val_score(rf,x,y,cv=10,scoring='f1_weighted')
#accuracy_rf = cross_val_score(rf,x_train,y_train,cv=10, scoring='accuracy')
accuracy_rf = cross_val_score(rf,x,y,cv=10, scoring='accuracy')
accuracy_rf=accuracy_rf*100
print(f_measure_rf)
print(accuracy_rf)

#MLP CLASSIFIER
#mlp = MLPClassifier(random_state=0,alpha=1, max_iter=1000)
sc=StandardScaler()

scaler = sc.fit(x_train)
x_train_s = scaler.transform(x_train)
x_test_s= scaler.transform(x_test)

mlp = MLPClassifier(hidden_layer_sizes=(150,100,50),activation='relu', max_iter = 2000, solver='adam', random_state=1)
mlp.fit(x_train_s, y_train)
y_pred = mlp.predict(x_test_s)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))

# 10-fold cross-validation for MLP
f_measure_mlp = cross_val_score(mlp,x_train_s,y_train,cv=10,scoring='f1_weighted')
accuracy_mlp  = cross_val_score(mlp,x_train_s,y_train,cv=10, scoring='accuracy')
accuracy_mlp=accuracy_mlp*100
print(f_measure_mlp)
print(accuracy_mlp)

#Create Table 1 - KNN
columns=np.transpose([f_measure_knn,accuracy_knn])
dknn=pd.DataFrame(columns,columns=['f_measure_knn','accuracy_knn'])
col_1=np.transpose(f_measure_knn)
col_2=np.transpose(accuracy_knn)
print('f_measure_knn',col_1)
print('accuracy_knn',col_2)

#Create Table 2 - RF
columns=np.transpose([f_measure_rf,accuracy_rf])
drf=pd.DataFrame(columns,columns=['f_measure_rf','accuracy_rf'])
col_3=np.transpose(f_measure_rf)
col_4=np.transpose(accuracy_rf)
print('f_measure_rf',col_3)
print('accuracy_rf',col_4)

#Create Table 3 - MLP
columns=np.transpose([f_measure_mlp,accuracy_mlp])
dmlp=pd.DataFrame(columns,columns=['f_measure_mlp','accuracy_mlp'])
col_5=np.transpose(f_measure_mlp)
col_6=np.transpose(accuracy_mlp)
print('f_measure_mlp',col_5)
print('accuracy_mlp',col_6)

#export file

writer = pd.ExcelWriter('Baseline_D_'+'.xlsx', engine='xlsxwriter')
dknn.to_excel(writer,sheet_name='KNN')
drf.to_excel(writer,sheet_name='RF')
dmlp.to_excel(writer,sheet_name='MLP')
writer.save()
files.download(writer)