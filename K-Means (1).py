#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# ## Clustering-Divide the universities in to groups(Cluster)

# In[3]:


Univ1 = pd.read_csv("Universities.csv")
Univ1


# In[6]:


Univ1.info()


# In[8]:


Univ1.describe()


# In[10]:


Univ1.head()


# ### Standardization of the data

# In[13]:


#Read all numeric columns in to Univ1
Univ1 = Univ1.iloc[:,1:]


# In[15]:


Univ1


# In[17]:


cols = Univ1.columns


# In[43]:


# Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ1_df = pd.DataFrame(scaler.fit_transform(Univ1), columns = cols)
scaled_Univ1_df


# In[45]:


#Build 3 clusters using KMeans cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ1_df)


# In[23]:


# Print the cluster labels
clusters_new.labels_


# In[35]:


set(clusters_new.labels_)


# In[47]:


Univ1['clusterid_new'] = clusters_new.labels_


# In[49]:


#Assign clusters to the Univ data set
Univ1.sort_values(by = "clusterid_new")


# In[51]:


# Use groupby() to find aggregated(mean) values in each cluster
Univ1.iloc[:,1:].groupby("clusterid_new").mean()


# ### Observations
# - Cluster 2 appears to be the top rated universities cluster as the cut off score, Top 10, SfRatio parameter mean values are highest
# - Cluster 1 appears to occupy the middle level rated universities
# - Cluster 0 comes as the lower level rated universities

# In[54]:


Univ1[Univ1['clusterid_new']==0]


# ### Finding potimal k values using elbow plot

# In[61]:


wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(scaled_Univ1_df)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:




