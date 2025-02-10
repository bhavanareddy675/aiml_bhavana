#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# ## Clustering-Divide the universities in to groups(Cluster)

# In[27]:


Univ1 = pd.read_csv("Universities.csv")
Univ1


# In[29]:


Univ1.info()


# In[33]:


Univ1.describe()


# In[35]:


Univ1.head()


# ### Standardization of the data

# In[39]:


#Read all numeric columns in to Univ1
Univ1 = Univ1.iloc[:,1:]


# In[41]:


Univ1


# In[43]:


cols = Univ1.columns


# In[47]:


# Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1), columns = cols)
scaled_Univ_df


# In[ ]:




