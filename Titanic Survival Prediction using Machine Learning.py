#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Collection & Processing

# In[ ]:


# load the data from csv file to Pandas DataFrame
titanic_data = pd.read_csv('train.csv')


# In[ ]:


# printing the first 5 rows of the dataframe
titanic_data.head()


# In[ ]:


# number of rows and Columns
titanic_data.shape


# In[ ]:


# getting some informations about the data
titanic_data.info()


# In[ ]:


# check the number of missing values in each column
titanic_data.isnull().sum()


# Handling the Missing values

# In[ ]:


# drop the "Cabin" column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)


# In[ ]:


# replacing the missing values in "Age" column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[ ]:


# finding the mode value of "Embarked" column
print(titanic_data['Embarked'].mode())


# In[ ]:


print(titanic_data['Embarked'].mode()[0])


# In[ ]:


# replacing the missing values in "Embarked" column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# In[ ]:


# check the number of missing values in each column
titanic_data.isnull().sum()


# Data Analysis

# In[ ]:


# getting some statistical measures about the data
titanic_data.describe()


# In[ ]:


# finding the number of people survived and not survived
titanic_data['Survived'].value_counts()


# Data Visualization

# In[ ]:


sns.set()


# In[ ]:


# making a count plot for "Survived" column
sns.countplot('Survived', data=titanic_data)


# In[ ]:


titanic_data['Sex'].value_counts()


# In[ ]:


# making a count plot for "Sex" column
sns.countplot('Sex', data=titanic_data)


# In[ ]:


# number of survivors Gender wise
sns.countplot('Sex', hue='Survived', data=titanic_data)


# In[ ]:


# making a count plot for "Pclass" column
sns.countplot('Pclass', data=titanic_data)


# In[ ]:


sns.countplot('Pclass', hue='Survived', data=titanic_data)


# Encoding the Categorical Columns

# In[ ]:


titanic_data['Sex'].value_counts()


# In[ ]:


titanic_data['Embarked'].value_counts()


# In[ ]:


# converting categorical Columns

titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[ ]:


titanic_data.head()


# Separating features & Target

# In[ ]:


X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']


# In[ ]:


print(X)


# In[ ]:


print(Y)


# Splitting the data into training data & Test data

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[ ]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# Logistic Regression

# In[ ]:


model = LogisticRegression()


# In[ ]:


# training the Logistic Regression model with training data
model.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[ ]:


# accuracy on training data
X_train_prediction = model.predict(X_train)


# In[ ]:


print(X_train_prediction)


# In[ ]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[ ]:


# accuracy on test data
X_test_prediction = model.predict(X_test)


# In[ ]:


print(X_test_prediction)


# In[ ]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# In[ ]:




