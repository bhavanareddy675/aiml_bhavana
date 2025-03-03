#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[4]:


iris = pd.read_csv("iris.csv")
iris


# In[5]:


import seaborn as sns
counts = iris["variety"].value_counts()
sns.barplot(data = counts)


# In[6]:


iris.info()


# In[7]:


iris[iris.duplicated(keep= False)]


# # observations
# * There are 150 rows and 5 columns
# * There are no Null values
# * There is one duplicated row
# * The x-columns are sepal.length, sepal.width, petal.length and petal.widtl
# * All the x-columns are continuous
# * The y-column is "variety" which is categorical
# * There are three flower categories (classes)

# In[9]:


iris = iris.drop_duplicates(keep='first')


# In[10]:


iris[iris.duplicated]


# In[21]:


iris = iris.reset_index(drop=True)
iris


# In[23]:


labelencoder = LabelEncoder()
iris.iloc[:, -1] = labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[25]:


# Check the data types after label encoding
iris.info()


# ## Observation
# - The target column('variety') is still object type.it needs to be converted to numeric(int)

# In[30]:


iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[36]:


X=iris.iloc[:,0:4]
Y=iris['variety']


# In[38]:


x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3, random_state = 1)
x_train


# In[40]:


x_train


# ## Building Decision Tree Classifier using Entropy Criteria

# In[45]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth = None)
model.fit(x_train,y_train)


# In[50]:


# Plot the decision tree
plt.figure(dpi=1200)
tree.plot_tree(model);


# In[52]:


fn=['sepal length (cm)','sepal width(cm)','petal length (cm)','petal width (cm)']
cn=['setosa','versicolor','virginica']
plt.figure(dpi=1200)
tree.plot_tree(model,feature_names = fn, class_names=cn, filled = True);


# In[ ]:




