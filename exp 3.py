#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[6]:


df=pd.read_csv(r"C:\Users\chiga\Downloads\titanic_dataset.csv")


# In[7]:


df.info()


# In[8]:


df.head()


# In[9]:


df.isnull().sum()


# In[10]:


df.drop("Cabin",axis=1,inplace=True)


# In[11]:


df.info()


# In[12]:


df.isnull().sum()


# In[13]:


df["Age"]=df["Age"].fillna(df["Age"].median())


# In[14]:


df.boxplot()


# In[15]:


df.isnull().sum()


# In[16]:


df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode())


# In[17]:


df.isnull().sum()


# In[18]:


df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])


# In[19]:


df["Embarked"].value_counts()


# In[20]:


df["Pclass"].value_counts()


# In[21]:


df["Survived"].value_counts()


# In[22]:


sns.countplot(x="Survived",data=df)


# In[23]:


sns.countplot(x="Pclass",data=df)


# In[24]:


sns.countplot(x="Sex",data=df)


# In[25]:


df.info()


# In[26]:


sns.displot(df["Fare"])


# In[27]:


sns.countplot(x="Pclass",hue="Survived",data=df)


# In[28]:


sns.countplot(x="Sex",hue="Survived",data=df)


# In[29]:


sns.displot(df[df["Survived"]==0]["Age"])


# In[30]:


sns.displot(df[df["Survived"]==1]["Age"])


# In[31]:


pd.crosstab(df["Pclass"],df["Survived"])


# In[32]:


pd.crosstab(df["Sex"],df["Survived"])


# In[33]:


df.corr()


# In[34]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:




