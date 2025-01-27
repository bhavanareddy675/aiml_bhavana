#!/usr/bin/env python
# coding: utf-8

# ### Exploratory Data Analysis 1

# In[7]:


#Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


data = pd.read_csv("data_clean.csv")
data


# In[9]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[10]:


data.info()


# In[11]:


# Data structure
print(type(data))
print(data.shape)


# In[12]:


data.shape


# In[14]:


#Drop duplicate column and unnamed column
data1 = data.drop(['Unnamed: 0',"Temp C"],axis =1)
data1


# In[21]:


# convert the Month column data type to integer data type
data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[23]:


data1.rename({'Solar: R':'Solar'}, axis = 1,inplace = True)
data1


# In[25]:


#Display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[26]:


#Visualize data1 missing values

cols = data1.columns
colours = ['lavender', 'pink']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)


# In[28]:


# Find the mean and median values of each numeric column
# Imppute of missing value with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[29]:


# replace the ozone missing values with median value
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[30]:


median_solar = data1["Solar.R"].median()
mean_solar = data1["Solar.R"].mean()
print("Median of Solar.R: ", median_solar)
print("Mean of Solar.R: ", mean_solar)


# In[31]:


data1['Solar.R'] = data1['Solar.R'].fillna(median_solar)
data1.isnull().sum()


# In[32]:


median_wind = data1["Wind"].median()
mean_wind = data1["Wind"].mean()
print("Median of Wind: ", median_wind)
print("Mean of Wind: ", mean_wind)


# In[33]:


data1['Wind'] = data1['Wind'].fillna(median_wind)
data1.isnull().sum()


# In[34]:


# Find the mode values of categorical coulumn(weather)
print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[35]:


# Impute missing values (Replace NaN with mode etc.) of "weather" using fillna()
data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[36]:


print(data1["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[37]:


data1["Month"] = data1["Month"].fillna(mode_weather)
data1.isnull().sum()


# In[39]:


data1.tail()


# In[52]:


#Reset the index column
data1.reset_index(drop=True)


# In[56]:


#Create a figure with two subplots ,stacked vertically
fig, axes=plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios':[1, 3]})
#Plot the boxplot in the first (top) subplot
sns.boxplot(data=data1["Ozone"], ax=axes[0],color='skyblue',width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")

#Plot the histogram with KDE Curve in the second (bottom) subplot
sns.histplot(data1["Ozone"], kde=True, ax=axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")
 
#Adjust layout for better spacing
plt.tight_layout()

#Show the plot
plt.show()


# ## Observations
# - The ozone column has extreme values beyond 81 as seen from plot
# - The same is confirmed from the below right-skewed histogram

# In[61]:


fig, axes=plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios':[1, 3]})
#Plot the boxplot in the first (top) subplot
sns.boxplot(data=data1["Solar"], ax=axes[0],color='skyblue',width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")

#Plot the histogram with KDE Curve in the second (bottom) subplot
sns.histplot(data1["Solar"], kde=True, ax=axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("SolaLevels")
axes[1].set_ylabel("Frequency")
 
#Adjust layout for better spacing
plt.tight_layout()

#Show the plot
plt.show()

