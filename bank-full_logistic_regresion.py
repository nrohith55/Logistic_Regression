# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:16:42 2020

@author: Rohith
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Logistic_Regression\\bank-full.csv",sep=";")

df=pd.get_dummies(df,columns=['job','marital','education','default','housing','loan','contact','month','poutcome','y'],drop_first=True)

df.shape

df.isnull().sum()

sns.countplot(x="age",data=df,palette='hls')

sns.barplot(x="age",y="balance",data=df,palette='hls')

pd.crosstab(df.age,df.balance).plot(kind='bar')

#Modelbuilding

X=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]]
Y=df.iloc[:,42]
classifier=LogisticRegression()
classifier.fit(X,Y)
y_pred=classifier.predict(X)
print(y_pred)
df["y_pred"]=y_pred

classifier.coef_
classifier.predict_proba(X)

y_prob=pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
new_df=pd.concat([df,y_prob],axis=1)

confusion_matrix=confusion_matrix(Y,y_pred)

from sklearn.metrics import accuracy_score

accuracy_score(Y,y_pred)
