# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:43:27 2020

@author: Rohith
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

df=pd.read_csv("E:\Data Science\Assignments\Python code\Logistic_Regression\\AFFAIRS.csv")
df=pd.get_dummies(df,columns={"gender","children"},drop_first=True)
df.head()
df.columns
sns.pairplot(df)
df.isnull().sum()
df.shape

#Dataset vizualization

sns.countplot(x='age',data=df,palette='hls')
sns.countplot(x='yearsmarried',data=df,palette='hls')
sns.countplot(x='religiousness',data=df,palette='hls')
sns.countplot(x='education',data=df,palette='hls')
sns.countplot(x='occupation',data=df,palette='hls')
sns.countplot(x='rating',data=df,palette='hls')

sns.boxplot(x="religiousness",y="age",data=df,palette='hls')
sns.boxplot(x="affairs",y="age",data=df,palette='hls')

pd.crosstab(df.affairs,df.age).plot(kind='bar')
pd.crosstab(df.affairs,df.yearsmarried).plot(kind='bar')

#Modelbuilding
X=df.iloc[:,[1,2,3,4,5,6,7,8]]
Y=df.iloc[:,0]
classifier=LogisticRegression()
classifier.fit(X,Y)
y_pred=classifier.predict(X)
df["y_pred"]=y_pred

classifier.coef_
classifier.predict_proba(X)
y_prob=pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
y_prob

new_data=pd.concat([df,y_prob],axis=1)

from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(Y,y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(Y, y_pred, normalize=True, sample_weight=None)
