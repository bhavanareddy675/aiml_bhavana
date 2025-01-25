#!/usr/bin/env python
# coding: utf-8

# ### Exploratory Data Analysis 1

# In[1]:


#Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
data


# In[7]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[9]:


data.info()


# In[11]:


# Data structure
print(type(data))
print(data.shape)


# In[13]:


data.shape


# In[15]:


#Drop duplicate column and unnamed column
data1 = data.drop(['Unnamed: 0',"Temp C"],axis =1)
data1


# In[17]:


# convert the Month column data type to integer data type
data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[25]:


data1.rename({'Solar: R':'Solar'}, axis = 1,inplace = True)
data1


# In[19]:


#Display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[33]:


#Visualize data1 missing values

cols = data1.columns
colours = ['lavender', 'pink']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)


# In[35]:


# Find the mean and median values of each numeric column
# Imppute of missing value with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[39]:


# replace the ozone missing values with median value
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[45]:


median_solar = data1["Solar.R"].median()
mean_solar = data1["Solar.R"].mean()
print("Median of Solar.R: ", median_solar)
print("Mean of Solar.R: ", mean_solar)


# In[47]:


data1['Solar.R'] = data1['Solar.R'].fillna(median_solar)
data1.isnull().sum()


# In[49]:


median_wind = data1["Wind"].median()
mean_wind = data1["Wind"].mean()
print("Median of Wind: ", median_wind)
print("Mean of Wind: ", mean_wind)


# In[53]:


data1['Wind'] = data1['Wind'].fillna(median_wind)
data1.isnull().sum()


# In[ ]:




