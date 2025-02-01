#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


# In[26]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# ### EDA

# In[29]:


data1.info()


# In[31]:


data1.isnull().sum()


# In[33]:


data1.describe()


# In[39]:


## Boxplot for daily column
plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"], vert = False)
plt.show()


# In[37]:


sns.histplot(data1['daily'], kde = True, stat='density')
plt.show()


# In[41]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Sunday Sales")
plt.boxplot(data1["sunday"], vert = False)
plt.show()


# In[43]:


sns.histplot(data1['sunday'], kde = True, stat='density')
plt.show()


# ### Observations
# - There are no missing values
# - The daily column values appears to be right-skewed
# - The sunday column values also appear to be right-skewed
# - There are two outliers in both daily column and also in sunday column as observed from the boxplots

# ### Scatter plot and Correlation Strength
# 

# In[47]:


x= data1["daily"]
y= data1["sunday"]
plt.scatter(data1["daily"], data["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[49]:


data1["daily"].corr(data1["sunday"])


# In[51]:


data1[["daily","sunday"]].corr()


# ### Observation
# - The relationship between x(daily) and y(sunday) is seen to be linear as seen from scatter plot
# - The correlation is strong positive with Pearson's correlation coefficient of 0.958254

# ## Fit a Linear Regression Model

# In[59]:


# Build regression model
import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[61]:


model1.summary()


# In[ ]:




