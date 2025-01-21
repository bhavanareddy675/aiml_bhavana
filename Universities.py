#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[7]:


#Mean value of SAT scorE
df = pd.read_csv("Universities.csv")
df


# In[9]:


np.mean(df["SAT"])


# In[11]:


np.median(df["SAT"])


# In[ ]:




