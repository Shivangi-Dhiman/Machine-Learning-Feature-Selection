from google.colab import files
uploaded = files.upload()

#Execute the above commande twice to Upload Dataset and ranking (which you got from Matlab)


#Execute the below commands to import missing libraries
!pip install git+https://github.com/jundongl/scikit-feature.git
pip install xlsxwriter

from google.colab import files
import pandas as pd
import numpy as np
from google.colab import drive
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn import svm
import warnings
warnings.simplefilter('ignore')
import xlsxwriter



df=pd.read_csv('D1.csv') ##### Dataset File
y = df.iloc[:,-1].values 
x = df.iloc[:,:-1].values 


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


read =pd.read_csv('D1_UDFS.csv',header=None) ##### Ranking File Name
read=read-1
scores = read.iloc[:,-1]





#RandomForestClassifier

no_features = 15 #####
rf = RandomForestClassifier()
rf= rf.fit(x_train[:, scores[0:no_features]], y_train)
y2_pred = rf.predict(x_test[:, scores[0:no_features]])
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y2_pred)))

#10-fold cross-validation for RF
f_measure_rf = cross_val_score(rf,x,y,cv=10,scoring='f1_weighted')
accuracy_rf  = cross_val_score(rf,x,y,cv=10, scoring='accuracy')
accuracy_rf = accuracy_rf*100

columnrf=np.transpose([f_measure_rf,accuracy_rf])
dfrf=pd.DataFrame(columnrf,columns=['f_measure_rf','accuracy_rf'])
col_1=np.transpose(f_measure_rf)
col_2=np.transpose(accuracy_rf)
print('f_measure_rf',col_1)
print('accuracy_rf',col_2)


#MLPClassifier

no_features = 20 #####
sc=StandardScaler()
scaler = sc.fit(x_train)
x_train_s = scaler.transform(x_train)
x_test_s= scaler.transform(x_test)
mlp = MLPClassifier(hidden_layer_sizes=(150,100,50),activation='relu', max_iter = 2000, solver='adam', random_state=1)
mlp = mlp.fit(x_train_s[:, scores[0:no_features]], y_train)
y2_pred = mlp.predict(x_test_s[:, scores[0:no_features]])
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y2_pred)))

# 10-fold cross-validation for MLP
f_measure_mlp = cross_val_score(mlp,x_train_s,y_train,cv=10,scoring='f1_weighted')
accuracy_mlp  = cross_val_score(mlp,x_train_s,y_train,cv=10, scoring='accuracy')
accuracy_mlp = accuracy_mlp*100

columnsmlp=np.transpose([f_measure_mlp,accuracy_mlp])
dfmlp=pd.DataFrame(columnsmlp,columns=['f_measure_mlp','accuracy_mlp'])
col_3=np.transpose(f_measure_mlp)
col_4=np.transpose(accuracy_mlp)
print('f_measure_MLP',col_3)
print('accuracy_MLP',col_4)


#KNN

no_features = 25 #####
knn = KNeighborsClassifier(n_neighbors=1)
knn = knn.fit(x_train[:, scores[0:no_features]], y_train)
y2_pred = knn.predict(x_test[:, scores[0:no_features]])
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y2_pred)))

# 10-fold cross-validation for KNN
f_measure_knn = cross_val_score(knn,x,y,cv=10,scoring='f1_weighted')
accuracy_knn  = cross_val_score(knn,x,y,cv=10, scoring='accuracy')
accuracy_knn = accuracy_knn*100

columnsknn=np.transpose([f_measure_knn,accuracy_knn])
dfknn=pd.DataFrame(columnsknn,columns=['f_measure_knn','accuracy_knn'])
col_5=np.transpose(f_measure_knn)
col_6=np.transpose(accuracy_knn)
print('f_measure_KNN',col_5)
print('accuracy_KNN',col_6)


#Saving into excel file
#library to install: pip install xlsxwriter

writer = pd.ExcelWriter('WithFS_DS_'+'.xlsx', engine='xlsxwriter')
dfrf.to_excel(writer,sheet_name='RF')
dfmlp.to_excel(writer,sheet_name='MLP')
dfknn.to_excel(writer,sheet_name='KNN')
writer.save()
files.download(writer)

#excel file saved with 3 sheets