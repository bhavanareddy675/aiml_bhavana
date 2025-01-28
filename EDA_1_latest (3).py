#!/usr/bin/env python
# coding: utf-8

# ### Exploratory Data Analysis 1

# In[4]:


#Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


data = pd.read_csv("data_clean.csv")
data


# In[6]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[7]:


data.info()


# In[8]:


# Data structure
print(type(data))
print(data.shape)


# In[9]:


data.shape


# In[10]:


#Drop duplicate column and unnamed column
data1 = data.drop(['Unnamed: 0',"Temp C"],axis =1)
data1


# In[11]:


# convert the Month column data type to integer data type
data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[12]:


data1.rename({'Solar: R':'Solar'}, axis = 1,inplace = True)
data1


# In[13]:


#Display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[14]:


#Visualize data1 missing values

cols = data1.columns
colours = ['lavender', 'pink']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)


# In[15]:


# Find the mean and median values of each numeric column
# Imppute of missing value with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[16]:


# replace the ozone missing values with median value
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[17]:


median_solar = data1["Solar.R"].median()
mean_solar = data1["Solar.R"].mean()
print("Median of Solar.R: ", median_solar)
print("Mean of Solar.R: ", mean_solar)


# In[28]:


data1['Solar.R'] = data1['Solar.R'].fillna(median_solar)
data1.isnull().sum()


# In[29]:


median_wind = data1["Wind"].median()
mean_wind = data1["Wind"].mean()
print("Median of Wind: ", median_wind)
print("Mean of Wind: ", mean_wind)


# In[30]:


data1['Wind'] = data1['Wind'].fillna(median_wind)
data1.isnull().sum()


# In[38]:


# Find the mode values of categorical coulumn(weather)
print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[40]:


# Impute missing values (Replace NaN with mode etc.) of "weather" using fillna()
data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[42]:


print(data1["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[44]:


data1["Month"] = data1["Month"].fillna(mode_weather)
data1.isnull().sum()


# In[46]:


data1.tail()


# In[48]:


#Reset the index column
data1.reset_index(drop=True)


# In[50]:


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

# In[57]:


#create a figure with two subplots, stacked vertically
fig, axes = plt.subplots(2,1,figsize=(8,16), gridspec_kw={'height_ratios':[1,3]})

#plot the boxplot in the first (top) subplot
sns.boxplot(data1['Solar.R'],ax=axes[0], color='skyblue', width=0.5, orient= 'h')
axes[0].set_title('Boxplot')
axes[0].set_xlabel('Solar.R Levels')

# Plot the histogram with KDE curve in the second (bottom) subplot
sns.histplot(data1["Solar.R"], kde=True, ax=axes[1], color='purple', bins=30)
axes [1] .set_title("Histogram with KDE")
axes [1] .set_xlabel("Solar.R Levels")
axes [1] .set_ylabel("Frequency")

#Adjust layout for better spacing
plt.tight_layout()

#show the plot
plt.show()


# In[61]:


sns.violinplot(data=data1["Solar.R"], color='lightgreen')
plt.title("Violin Plot")


# In[71]:


plt.figure(figure=(6,2))
plt.boxplot(data1["Ozone"], vert=False)


# In[67]:


# Extract outliers from boxplot for Ozone column
plt.figure(figure=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']] 


# #### Method 2 for outlier detection
# - Using mu+/-3*sigma limits(Standard deciation method)

# In[74]:


data1["Ozone"].describe()


# In[78]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]

for x in data1["Ozone"]:
    if((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# ### Observation
# - It is observed that only two outliers are identified using std method
# - In box plot method more no of outliers are identified
# - This is because the assumption of normality is not satisfied in this column

# #### Quantile-Quantile pl0ot for detection of outliers

# In[82]:


import scipy.stats as stats

#Create Q-Q plot
plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q Plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# ## Observation from Q-Q plot 
# -The data does not follow normal distribution as the data points are deviating significantly away from the red line.
# 
# -The data shows a right-skewed distribution and possible outliers

# In[ ]:




