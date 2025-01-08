#!/usr/bin/env python
# coding: utf-8

# In[1]:


num = 6
if num %2 == 0:
    print("even")
else:
    print("odd")


# In[2]:


print("even")if num %2 == 0 else print("odd")


# In[6]:


num = 3
result = "Positive" if num > 0 else ("Negative" if num < 0 else "zero")
print(result)


# ## List Comprehension

# In[7]:


L = [1, 9, 2, 10, 56, 89]
[2*x for x in L]


# In[8]:


[x for x in L if x%2 == 0]


# In[9]:


[x for x in L if x%2 != 0]


# In[12]:


# print the average value of the list of numbers using list comprehension
L = [1, 9, 2, 10, 56, 89]
sum([x for x in L])/len(L)


# In[13]:


#Dictionary comprehension
d1 = {'Ram':[70,71,98,100], 'Jhon':[56,98,67,65]}
d1


# In[16]:


{k:sum(v)/len(v) for k,v in d1.items()}

