#!/usr/bin/env python
# coding: utf-8

# In[1]:


greet = lambda name : print(f"Good Morning {name}!")


# In[2]:


greet("Bannuuu")


# In[3]:


product = lambda a,b,c : a*b*c


# In[5]:


product(7,3,2024)


# In[6]:


even = lambda L : [x for x in L if x%2 == 0]


# In[8]:


my_list = [100,3,9,38,43,56,20]
even(my_list)


# In[10]:


my_list = [100,3,9,38,43,56,20]
odd = lambda L : [x for x in L if x%2 != 0]
odd(my_list)


# ## Python Modules

# In[13]:


def mean_value(*n):
    sum = 0
    counter = 0
    for x in n:
        counter = counter +1
        sum += x
    mean = sum /counter
    return mean


# In[16]:


mean_value(25,46,84,97)


# In[17]:


def product(*n):
    result = 1
    for i in range(len(n)):
        result *= n[i]
    return result


# In[18]:


product(7*3*24)


# In[ ]:




