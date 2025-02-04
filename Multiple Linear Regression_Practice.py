#!/usr/bin/env python
# coding: utf-8

# #### Assumptions in Multilinear regression
# 1.Linearity:The relationship between the predictors (X) and the response variable (Y) is linear.
# 
# 2.Independence:Observations are independent of each other.
# 
# 3.Homoscedasticity:The residuals (Y - Y_hat) exhibit constant variance at all levels of the predictor.
# 
# 4.Normal Distribution of Errors:The residuals (errors) of the model are normally distributed.
# 
# 5.No multicollinearity: The independent variables should not be too highly correlated with each other.
# 

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[8]:


# Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[10]:


# Rearrange the columns
cars = pd.DataFrame(cars, columns=["Hp","VOL","SP","WT","MPG"])
cars.head()


# ### Description of columns
# - MPG: Milege of the car (Mile per Gallon) (This is Y-column to be predicted)
# - HP: Horse Power of the car (X! column)
# - VOL: Volume of the car (size) (X2 column)
# - SP: Top speed of the car(Miles per Hour)(X3 column)
# - WT: Weight of the car(Pounds) (X4 column) 

# ## EDA

# In[15]:


cars.info()


# In[17]:


# check for missing values
cars.isna().sum()


# ## Observations
# - There are no missing values
# - There are 81 observations (81 different cars data)
# - the data types of the columns are also relevant and valid

# In[ ]:




