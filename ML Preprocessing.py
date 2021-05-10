print('hello')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("C:\\ML\\loan prediction.csv")
dfPP=pd.read_csv("C:\\ML\\auto_mpg_seaborn.csv")

data1=data.copy()
data1.isnull().sum()
len(data1)

#drop rows
data1.dropna(inplace=True)
data1.dropna(subset=['LoanAmount'],inplace=True)

#fill with 0
data1.fillna(0,inplace=True)
data['LoanAmount'].fillna(0,inplace=True)

#mean
data1['LoanAmount'].fillna(data1['LoanAmount'].mean(),inplace=True)

#Median
data1['LoanAmount'].fillna(data1['LoanAmount'].median(),inplace=True)

#Mode
data1['Self_Employed'].value_counts()/len(data1['Self_Employed'])
mode1=data1['Self_Employed'].mode()
data1['Self_Employed'].fillna(mode1[0],inplace=True)

#Feature Eng
data['TotalIncome']=data['ApplicantIncome']+data['CoapplicantIncome']

#feature scaling
data_scale=data1[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']]

data_scale1=data_scale.fillna(0)
#z=(x-mean0/std)
sc=StandardScaler()
data2=sc.fit_transform(data_scale1)

#LabelEncoding
data1=data.copy()
data1['Married'].value_counts()
mode2=data1['Married'].mode()
data1['Married'].fillna(mode2[0],inplace=True)
le=LabelEncoder()
data1['Married']=le.fit_transform(data1['Married'])

#Dummy variables( one hot encoder)
dummies=pd.get_dummies(data1,columns=['Education','Dependents'])
dummies2=pd.get_dummies(data1,columns=['Education','Dependents'],drop_first=True)
