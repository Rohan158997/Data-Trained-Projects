#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm


# In[2]:


#import the datset to pandas Dataframe
df = pd.read_csv('LP.csv')


# In[3]:


df.head()


# As we have generated the dataset for the loan prediction we have got these 13 columns. There are some missing values in some columns as well. Few of the columns(Gender, Married,Education, self_Employed,Credit_history,property_area, Loan_status)have only 2 reults,While other columns result vary

# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


#We have total 614 rows and 13 columns, simply we can say there are 614 values for every single column


# In[7]:


df.isnull().sum()


# We will create one more column that is loan amountlog using loan amount detail than we will create histogram for this

# In[8]:


df['loanAmount_log'] = np.log(df['LoanAmount'])
df['loanAmount_log'].hist(bins=20)


# In[9]:


df.isnull().sum()


# In[10]:


#There are 22 missing values in our new column
df.head(3)


# In[11]:


#Lets create one more column Totalincome  in this we will add applicant income and coapplicant income
df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log']=np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)


# In[12]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace = True)
df['Married'].fillna(df['Married'].mode()[0],inplace = True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace = True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace = True)

df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
df.loanAmount_log = df.loanAmount_log.fillna(df.loanAmount_log.mean())

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace = True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace = True)

df.isnull().sum()


# In[13]:


df.shape


# In[36]:


X= df.iloc[:,np.r_[1:5,9:11,13:15]].values
Y= df.iloc[:,12].values


# In[37]:


X


# In[38]:


Y


# In[26]:


print("per of missing gender is %2f%%" %((df['Gender'].isnull().sum()/df.shape[0])*100))


# In[14]:


print("number of people who take loan as group by gender:")
print(df['Married'].value_counts())
sns.countplot(x='Married',data=df,  palette = 'Set1')


# In[15]:


print("number of people who take loan as group by gender:")
print(df['Gender'].value_counts())
sns.countplot(x='Gender',data=df,  palette = 'Set1')


# In[16]:


print("number of people who take loan as group by Dependents:")
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents',data=df,  palette = 'Set1')


# In[17]:


print("number of people who take loan as group by Self_Employed:")
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed',data=df,  palette = 'Set1')


# In[18]:


print("number of people who take loan as group by Loan Amount:")
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount',data=df,  palette = 'Set1')


# In[19]:


print("number of people who take loan as group by  Credit History:")
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History',data=df,  palette = 'Set1')


# In[54]:


# now we will seprate the data and label


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)

from sklearn.preprocessing import LabelEncoder
Labelencoder_X = LabelEncoder()


# In[40]:


for i in range(0,5):
    X_train[:,i]= Labelencoder_x.fit_transform(x_train[:,i])
    X_train[:,7]= Labelencoder_x.fit_transform(x_train[:,7])
    
X_train


# In[41]:


Labelencoder_Y = LabelEncoder()
Y_train= Labelencoder_Y.fit_transform(Y_train)

Y_train


# In[42]:


for i in range(0,5):
    X_test[:,i]= Labelencoder_X.fit_transform(X_test[:,i])
    X_test[:,7]= Labelencoder_X.fit_transform(X_test[:,7])
    
X_test


# In[43]:


Labelencoder_y = LabelEncoder()
Y_test= Labelencoder_Y.fit_transform(y_test)

Y_test


# In[48]:


from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[50]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, Y_train)


# In[52]:


from sklearn import metrics
Y_pred = rf_clf.predict(X_test)


print("acc ofrandom clf is", metrics.accuracy_score(Y_pred, Y_test))

Y_pred


# In[53]:


from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(X_train, Y_train)


# In[55]:


Y_pred = nb_clf.predict(X_test)
print("acc of naive bayes is ", metrics.accuracy_score(Y_pred, Y_test))


# In[56]:


from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train,Y_train)


# In[57]:


Y_pred = dt_clf.predict(X_test)
print ("acc of DT  is", metrics.accuracy_score(Y_pred, Y_test))


# In[58]:


Y_pred


# In[ ]:


GaussianNB gives the perfect model of Loan_status

