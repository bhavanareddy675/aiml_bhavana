#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Install mlxtend library
get_ipython().system('pip install mlxtend')


# In[5]:


# Import necessary Libraries

import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[7]:


titanic = pd.read_csv("Titanic.csv")
titanic


# In[9]:


titanic.info()


# ## Observations
# - As the columns are categorical, we can adopt one-hot-encoding
# - All columns are object data type and categorical in nature
# - There are no null values

# In[19]:


# plot a bar chart to visualize the category of class on the ship
counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[29]:


# Perform one-hot encoding on categorical columns
df = pd.get_dummies(titanic,dtype=int)
df.head()


# In[31]:


df.info()


# ## Apriori Algorithm

# In[27]:


# Apply APriori algorithmto get itemset combinations
frequent_itemsets = apriori(df, min_support = 0.05, use_colnames=True, max_len=None)
frequent_itemsets


# In[33]:


frequent_itemsets.info()


# In[35]:


#Generate association rules with metrics
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules


# In[37]:


rules.sort_values(by='lift', ascending = False)


# In[39]:


rules.sort_values(by='lift', ascending = False).head(20)


# ### Conclusion
# - Adult Females travelling in 1st class survived most

# In[44]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[ ]:




