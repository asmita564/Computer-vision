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
from sklearn.preprocessing import MinMaxScaler



from sklearn.metrics import (
accuracy_score, confusion_matrix, classification_report,
roc_auc_score, roc_curve, auc,
plot_confusion_matrix, plot_roc_curve
)


pip install xgboost
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


pip install tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

pip install --upgrade tensorflow==2.0.0-beta1


plt.style.use("fivethirtyeight")
sns.set_style('whitegrid')
%matplotlib inline



pd.set_option('display.float', '{:.2f}'.format)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)

## importing 
os.chdir("D:\SDSOS\Summer")
ld=pd.read_csv("lending_club_loan_two.csv")
s
#EDA##
#ld.describe()
#ld.info()
# missing values in title emp_title pub_rec_bankruptcies revo_util emp_length emp_title
#sns.countplot(ld.loan_status)

#plt.figure(figsize=(12, 8))
#sns.heatmap(ld.corr(), annot=True, cmap='viridis')

#ld.groupby(by='loan_status')['loan_amnt'].describe()

data=ld.loc[ld['annual_inc']>0]

data.describe()

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
#data.drop('sub_grade',axis=1,inplace=True)
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
data.zip_code.value_counts()
data.drop('address',axis=1,inplace=True)
data.drop('flag',axis=1,inplace=True)
data.drop('earliest_cr_line',axis=1,inplace=True)
dummies=['grade','verification_status','purpose','initial_list_status','application_type','home_ownership','term']


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
data.describe()




## oversampling


# check version number
#pip install "imblearn"
import imblearn
print(imblearn.__version__)

print(imblearn.__version__)
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')

## train test split
X = data.drop('loan_status', axis=1)
Y = data.loan_status
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_over, Y_over = oversample.fit_resample(X, Y)
Y_over.describe()
from collections import Counter
Counter(Y_train)
Counter(Y_test)
X_train, X_test, Y_train, Y_test = train_test_split(X_over, Y_over, test_size=0.25, random_state=15, stratify=Y_over)


## logistic Regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

logreg=LogisticRegression()
logreg.fit(X_train,Y_train)
logreg.score(X_test,Y_test)

y_pred=logreg.predict(X_test)
from sklearn.metrics import confusion_matrix
matrix= confusion_matrix(Y_test,y_pred)
print(matrix)
plt.figure(figsize=(15, 10))
sns.countplot(data.loan_status)
plt.savefig('Loan status bar chart.png', dpi=150)


#graphics
plt.figure(figsize=(12, 20))

plt.figure(figsize=(15, 20))
data[data["loan_status"] == "Fully Paid"]["installment"].hist(bins=20, color='blue', label='loan_status = Fully Paid', alpha=0.6)
data[data["loan_status"] == "Charged Off"]["installment"].hist(bins=20, color='red', label='loan_status = Charged Off', alpha=0.6)
plt.legend()
plt.xlabel("installment")
plt.savefig('Loan status and installment1.png', dpi=150)

#plt.subplot(4, 2, 2)
plt.figure(figsize=(15, 20))
data[data["loan_status"] == "Fully Paid"]["loan_amnt"].hist(bins=20, color='blue', label='loan_status = Fully Paid', alpha=0.6)
data[data["loan_status"] == "Charged Off"]["loan_amnt"].hist(bins=20, color='red', label='loan_status = Charged Off', alpha=0.6)
plt.legend()
plt.xlabel("loan_amnt")
plt.savefig('Loan status and loan amount1.png', dpi=150)

plt.figure(figsize=(15, 20))
#plt.subplot(4, 2, 3)
sns.scatterplot(x='loan_amnt', y='installment', data=data)
plt.savefig('loan amount and installment scatterplot.png', dpi=150)

#plt.subplot(4, 2, 4)
plt.figure(figsize=(15, 10))
sns.boxplot(x='loan_status', y='loan_amnt', data=data)
plt.savefig('loan status vs loan amount box plot.png', dpi=200)
#plt.subplot(2, 2, 1)
grade = sorted(data.grade.unique().tolist())


plt.figure(figsize=(15, 10))
sns.countplot(x='grade', data=data, hue='loan_status', order=grade)
plt.savefig('loan status and grade count.png', dpi=200)
#plt.subplot(2, 2, 2)
plt.figure(figsize=(15, 10))
sub_grade = sorted(data.sub_grade.unique().tolist())
g = sns.countplot(x='sub_grade', data=data, hue='loan_status', order=sub_grade)
g.set_xticklabels(g.get_xticklabels(), rotation=90);
plt.savefig('loan status and subgrade distribution.png', dpi=200)

df = data[(data.grade == 'F') | (data.grade == 'G')]

plt.figure(figsize=(15, 10))

#plt.subplot(2, 2, 1)
grade = sorted(df.grade.unique().tolist())
sns.countplot(x='grade', data=df, hue='loan_status', order=grade)
plt.savefig('loan status vs grade f and G.png', dpi=200)
#plt.subplot(2, 2, 2)
plt.figure(figsize=(15, 10))
sub_grade = sorted(df.sub_grade.unique().tolist())
sns.countplot(x='sub_grade', data=df, hue='loan_status', order=sub_grade)
plt.savefig('loan status vs F and G subgrade.png', dpi=200)



plt.figure(figsize=(15, 10))

#plt.subplot(4, 2, 1)
sns.countplot(x='term', data=data, hue='loan_status')
plt.savefig('loan status vs loan term.png', dpi=200)
plt.figure(figsize=(15, 20))


#data1=data[(data.home_ownership == "RENT") or (data.home_ownership == "OWN") or(data.home_ownership == "MORTGAGE")]
data1 = data[data["home_ownership"].isin(["RENT","OWN","MORTGAGE"])]
data1["home_ownership"].value_counts()

#plt.subplot(4, 2, 2)
plt.figure(figsize=(15, 10))
sns.countplot(x='home_ownership', data=data1, hue='loan_status')
plt.savefig('loan status vs loan homeownership.png', dpi=200)


plt.figure(figsize=(15, 10))
#plt.subplot(4, 2, 3)
sns.countplot(x='verification_status', data=data, hue='loan_status')
plt.savefig('loan status vs verification.png', dpi=200)


plt.figure(figsize=(15, 10))
#plt.subplot(4, 2, 4)
g = sns.countplot(x='purpose', data=data, hue='loan_status')
g.set_xticklabels(g.get_xticklabels(), rotation=90);
plt.savefig('loan status vs loan purpose.png', dpi=200)