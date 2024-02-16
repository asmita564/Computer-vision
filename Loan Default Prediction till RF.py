# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 12:26:29 2021

@author: dhadi
"""
## libraries
import os 
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split, RandomizedSearchCV
        



from sklearn.metrics import (
accuracy_score, confusion_matrix, classification_report,
roc_auc_score, roc_curve, auc,
plot_confusion_matrix, plot_roc_curve
)


#pip install xgboost
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


#pip install tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

#pip install --upgrade tensorflow==2.0.0-beta1


plt.style.use("fivethirtyeight")
sns.set_style('whitegrid')
%matplotlib inline



pd.set_option('display.float', '{:.2f}'.format)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)

## importing 
#os.chdir("D:\SDSOS\Summer")
#D:\SDSOS\Summer
ld=pd.read_csv("D:\SDSOS\Summer\lending_club_loan_two.csv")
ld.drop(ld.columns[[28]], axis = 1, inplace = True)
ld.drop(ld.columns[[27]], axis = 1, inplace = True)

#EDA##
#ld.describe()
#ld.info()
# missing values in title emp_title pub_rec_bankruptcies revo_util emp_length emp_title
#sns.countplot(ld.loan_status)

#plt.figure(figsize=(12, 8))
#sns.heatmap(ld.corr(), annot=True, cmap='viridis')

#ld.groupby(by='loan_status')['loan_amnt'].describe()

data=ld.loc[ld['annual_inc']>0]

#data.describe()

## emp title nikalna hain
## emp title bhi
## code same roundinf off extra work


## data preprocessing
for column in data.columns:
    if data[column].isna().sum() != 0:
        missing = data[column].isna().sum()
        portion = (missing / data.shape[0]) * 100
        print(f"'{column}': number of missing values '{missing}' ==> '{portion:.3f}%'")

## missing value treatments
## removing  emp title
data.drop('emp_title', axis=1, inplace=True)

## removing emp length
data.drop('emp_length', axis=1, inplace=True)

## removing issue date
data.drop('issue_d', axis=1, inplace=True)

## removing title
data.drop('title', axis=1, inplace=True)
data.drop('revol_util',axis=1,inplace=True)
data.drop('pub_rec_bankruptcies',axis=1,inplace=True)
data.drop('sub_grade',axis=1,inplace=True)

#imputing mort_acc
total_acc_avg = data.groupby(by='total_acc').mean().mort_acc

def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc].round()
    else:
        return mort_acc
data['mort_acc'] = data.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

# checking again for missing values
for column in data.columns:
    if data[column].isna().sum() != 0:
        missing = data[column].isna().sum()
        portion = (missing / data.shape[0]) * 100
        print(f"'{column}': number of missing values '{missing}' ==> '{portion:.3f}%'")
        
## getting pincodes from the address column

data['zip_code'] = data.address.apply(lambda x: x[-5:])
#data.zip_code.value_counts()
data.drop('address',axis=1,inplace=True)
#data.drop('flag',axis=1,inplace=True)
data.drop('earliest_cr_line',axis=1,inplace=True)
dummies=['grade','verification_status','purpose','initial_list_status','application_type','home_ownership','term','zip_code']

 
data= pd.get_dummies(data,columns=dummies,drop_first=True)


## removing duplicates from the dataset
#print(f"Data shape: {data.shape}")  

 # Remove duplicate Features
 #data = data.T.drop_duplicates()
 #data = data.T

 #Remove Duplicate Rows
 #data.drop_duplicates(inplace=True)

#print(f"Data shape: {data.shape}")
data["loan_status"].replace({"Fully Paid":"0","Charged Off":"1"},inplace=True)
#data.describe()




## oversampling


# check version number
#pip install "imblearn"
pip install imblearn
import imblearn
print(imblearn.__version__)

#print(imblearn.__version__)
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')

## correlation graph
import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
#corrmat = ld.corr()
#top_corr_features = corrmat.index
#plt.figure(figsize=(20,20))
#plot heat map
#g=sns.heatmap(ld[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#plt.savefig('heatmap ld data', dpi=150)


## train test split
X1=data.iloc[:,[0,1,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]]
Y=data.iloc[:,4]
scaler = MinMaxScaler()
X = scaler.fit_transform(X1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=70, stratify=Y)

from collections import Counter
Counter(Y)
Counter(Y_train)
Counter(Y_test)

X_train_over, Y_train_over = oversample.fit_resample(X_train, Y_train)
Y_train_over.describe()
Counter(Y_train_over)

## logistic Regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix




X2=data.iloc[:,[0,1,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]]
Y=data.iloc[:,4]
scaler = MinMaxScaler()
X_log = scaler.fit_transform(X2)



X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X_log, Y, test_size=0.25, random_state=70, stratify=Y)

from collections import Counter
Counter(Y)
Counter(Y_train)
Counter(Y_test)

X_train_over1, Y_train_over1 = oversample.fit_resample(X_train1, Y_train1)
Y_train_over.describe()
Counter(Y_train_over)
logreg=LogisticRegression(random_state=70,  max_iter=1000)
logreg.fit(X_train_over1,Y_train_over1)
logreg.score(X_test,Y_test)



y_predtest=logreg.predict(X_test1)
y_predtrain=logreg.predict(X_train_over1)
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test1,y_predtest)
confusion_matrix(Y_train_over1,y_predtrain)
print(matrix)

print(classification_report(Y_test1, y_predtest))
print(classification_report(Y_train_over1, y_predtrain))

disp = plot_roc_curve(logreg, X_train_over1, Y_train_over1)
disp = plot_roc_curve(logreg, X_test1, Y_test1)

disp = plot_confusion_matrix(logreg, X_test1, Y_test1, 
                             cmap='Blues', values_format='d', 
                             display_labels=[ 'Fully-Paid','Default',])

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

plot_roc_cur

#
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X_train_over,Y_train_over)
print(model.feature_importances_)
print(data.columns)
plt.figure(figsize=(15, 20))
feat_importances = pd.Series(model.feature_importances_, index=X1.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.title("Top 15 important features")
plt.show()








def print_score(true, pred, train=True):
    if train:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
        
    elif train==False:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
        
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)
from sklearn.model_selection import RandomizedSearchCV

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

#rf = RandomForestClassifier()
#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
#rf_random.fit(X_train,Y_train)
#rf_random.best_params_
#rf_random.best_score_
#{'n_estimators': 700,
 ##'min_samples_split': 15,
# 'min_samples_leaf': 1,
 #'max_features': 'auto',
 #'max_depth': 20}#



#WITH OVERSAMPLING
#ACCURACY : 88.36%
rf_clf = RandomForestClassifier( n_estimators=700,min_samples_split=15,min_samples_leaf=1,max_features="auto",max_depth=20)
rf_clf.fit(X_train_over, Y_train_over)
y_train_pred = rf_clf.predict(X_train_over)
y_test_pred = rf_clf.predict(X_test)

print_score(Y_train_over, y_train_pred, train = True)
print_score(Y_test, y_test_pred, train = False)

disp = plot_confusion_matrix(rf_clf, X_test, Y_test, 
                             cmap='Blues', values_format='d', 
                             display_labels=[ 'Fully-Paid''Default',])

disp = plot_roc_curve(rf_clf, X_test, Y_test)

rf_clf.feature_importances_

feat_importancesrf = pd.Series(rf_clf.feature_importances_, index=X1.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.title("Top 15 important features")
plt.show()
## 88.41




#WITHOUT OVERSAMPLING
#ACCURACY : 88.83%
rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, Y_train)
y_train_pred = rf_clf.predict(X_train)
y_test_pred = rf_clf.predict(X_test)

print_score(Y_train, y_train_pred, train = True)
print_score(Y_test, y_test_pred, train = False)
    
# 88.84
#data.to_csv("C:\Aditi\My Work\Projects\cleandata.csv")

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

classifier=xgboost.XGBClassifier()
## Hyper Parameter Optimization

param       s={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30,0.40,0.50 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1,2,4,6, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
  
from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X_train,Y_train)
timer(start_time)

random_search.best_estimator_
random_search.best_params_


classifier=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bytree=0.4, gamma=0.3, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.2, max_delta_step=0, max_depth=6,
              min_child_weight=4, missing=None, monotone_constraints='()',
              n_estimators=100, n_jobs=1, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None,eval_metric='logloss')

#from sklearn.model_selection import cross_val_score
#score=cross_val_score(classifier,X_train_over,Y_train_over,cv=10)
#score
#score.mean()
#Y_train.type()
classifier.fit(X_train1,Y_train1)
y_train_pred = classifier.predict(X_train_over)
y_test_pred = classifier.predict(X_test)

print_score(Y_train_over, y_train_pred, train=True)
print_score(Y_test, y_test_pred, train=False)


disp = plot_confusion_matrix(classifier, X_test, Y_test, 
                             cmap='Blues', values_format='d', 
                             display_labels=[ 'Default','Fully-Paid',])

disp=plot_roc_curve(classifier, X_test, Y_test)
plot_roc_curve(rf_clf, X_test, Y_test,ax=disp.ax_)
plot_roc_curve(knn, X_test, Y_test,ax=disp.ax_)
plot_roc_curve(logreg, X_test1, Y_test1,ax=disp.ax_)

## feature importance
feat_importancesxg = pd.Series(classifier.feature_importances_, index=X1.columns)
feat_importancesxg.nlargest(15).plot(kind='barh')
plt.title("Top 15 important features")
plt.show()
 
## permutation based feature importance
#perm_importance = permutation_importance(rf, X_test, y_test)



X_test=X_test.iloc[:,[0,1,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]]
## shap
pip install shap
import shap 
#shap for xgboost 
explainerxg = shap.TreeExplainer(classifier)
shap_values = explainerxg.shap_values(X_test1)
shap.nsmallest(5)
shap.summary_plot(shap_values, X_test1, plot_type="bar",feature_names=X1.columns)
shap.summary_plot(shap_values, X_test1, feature_names=X1.columns)
shap.summary_plot(shap_values, X_test)




# shap for random forest
explainerrf = shap.TreeExplainer(rf_clf)
shap_values = explainerrf.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test)
 

  

#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(random_state = 42)
#from pprint import pprint
# Look at parameters used by our current forest
#print('Parameters currently in use:\n')
#pprint(rf.get_params())
   
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train_over, Y_train_over)
print(knn.score(X_test, Y_test))

y_predknntest=knn.predict(X_test)
y_predknntrain=knn.predict(X_train_over)
print(classification_report(Y_train_over,y_predknntrain))
print(classification_report(Y_test,y_predknntest))



from sklearn.model_selection import RandomizedSearchCV
k=np.random.randint(1,50,60)
params={'n_neighbors': k}
knn.get_params()
random_search=RandomizedSearchCV(knn,params,n_iter=5,cv=5,n_jobs=-1,verbose=0)
random_search.fit(X_train_over,Y_train_over)
