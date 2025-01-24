#!/usr/bin/env python
# coding: utf-8

# ### Exploratory Data Analysis 1

# In[2]:


#Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data = pd.read_csv("data_clean.csv")
data


# In[6]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[8]:


data.info()


# In[10]:


# Data structure
print(type(data))
print(data.shape)


# In[12]:


data.shape


# In[22]:


#Drop duplicate column and unnamed column
data1 = data.drop(['Unnamed: 0',"Temp C"],axis =1)
data1


# In[26]:


# convert the Month column data type to integer data type
data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[ ]:




