#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# Data Collection

# In[8]:


# Loading the dataset as Pandas dataframe
wine_dataset = pd.read_csv('redwine.csv')
wine_dataset.shape


# In[12]:


#first five rows of the dataset
wine_dataset.sample(5)


# In[14]:


# checking missing values
wine_dataset.isnull().sum()


# we don't have any missing value so we can proceed with our analysis

# Data Analysis and Visualization

# In[15]:


wine_dataset.describe()


# In[16]:


# number of values for each quality 
sns.catplot(x='quality', data = wine_dataset, kind = 'count')


# In[17]:


# volatile acidity vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='volatile acidity', data =wine_dataset)


# In[ ]:


Volatile acidity is inversionaly propotional to quality


# In[18]:


# citric acid vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='citric acid', data =wine_dataset)


# citric acid content is directly propotional to quality 

# In[19]:


# residual sugar vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='residual sugar', data =wine_dataset)


# In[57]:


# chlorides vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='chlorides', data =wine_dataset)


# In[58]:


#larger the quantity of chloride bad is the quality of wine


# In[60]:


# free sulphur dioxide vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality', y='free sulfur dioxide', data =wine_dataset)


# Corelation

# In[20]:


correlation = wine_dataset.corr()


# In[25]:


#constructing a heat map to understand correlationbetween columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square= True, fmt = '.1f', annot=True, annot_kws={'size':8}, cmap='Greens')


# In[26]:


# Alcohol, Sulphates, citric acid and fixed acidity are directly propotional to the quality of wine
#  volatile acid, chlorides,free sulphur dioxide, Total sulphur dioxide,density, ph are inversionaly propotional to quality
# Residual sugar content is almost the same in all  the wines, so it will not impact the Quality analysis


# Data Preprocessing

# In[33]:


# seprate the data and label
X = wine_dataset.drop('quality', axis =1)


# In[34]:


print(X)


# Label Binarization

# In[35]:


Y = wine_dataset['quality'].apply(lambda y_value: 1if y_value>=7 else 0)
print(Y)


# Train Test Data

# In[36]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)


# In[37]:


print(Y.shape, Y_train.shape, Y_test.shape)


# Model Training:
# Random Forest classifier

# In[38]:


model = RandomForestClassifier()


# In[39]:


model.fit(X_train, Y_train)


# Model  Evaluation

# In[40]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[42]:


print('Accuracy : ', test_data_accuracy)


# out of 100 values our model can predict aarounnd 93 correct values, which is really really good 

# Building a Pedictive System

# In[53]:


input_data = (7.4,0.66,0.0,1.8,0.075,13.0,40.0,0.9978,3.51,0.56,9.4)


# In[54]:


#changing the input datatyp to numpy array
input_data_as_numpy_array = np.asarray(input_data)


# In[55]:


#rehaping the data as we are  predicting the level for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[56]:


prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==1):
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')


# In[ ]:





# In[ ]:




