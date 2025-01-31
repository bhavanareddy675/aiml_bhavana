#!/usr/bin/env python
# coding: utf-8

# ### Exploratory Data Analysis 1

# In[1]:


#Load the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
data


# In[3]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[4]:


data.info()


# In[5]:


# Data structure
print(type(data))
print(data.shape)


# In[6]:


data.shape


# In[7]:


#Drop duplicate column and unnamed column
data1 = data.drop(['Unnamed: 0',"Temp C"],axis =1)
data1


# In[8]:


# convert the Month column data type to integer data type
data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[9]:


data1.rename({'Solar: R':'Solar'}, axis = 1,inplace = True)
data1


# In[10]:


#Display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[11]:


#Visualize data1 missing values

cols = data1.columns
colours = ['lavender', 'pink']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)


# In[12]:


# Find the mean and median values of each numeric column
# Imppute of missing value with median
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[13]:


# replace the ozone missing values with median value
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[14]:


median_solar = data1["Solar.R"].median()
mean_solar = data1["Solar.R"].mean()
print("Median of Solar.R: ", median_solar)
print("Mean of Solar.R: ", mean_solar)


# In[15]:


data1['Solar.R'] = data1['Solar.R'].fillna(median_solar)
data1.isnull().sum()


# In[16]:


median_wind = data1["Wind"].median()
mean_wind = data1["Wind"].mean()
print("Median of Wind: ", median_wind)
print("Mean of Wind: ", mean_wind)


# In[17]:


data1['Wind'] = data1['Wind'].fillna(median_wind)
data1.isnull().sum()


# In[18]:


# Find the mode values of categorical coulumn(weather)
print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[19]:


# Impute missing values (Replace NaN with mode etc.) of "weather" using fillna()
data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[20]:


print(data1["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[21]:


data1["Month"] = data1["Month"].fillna(mode_weather)
data1.isnull().sum()


# In[22]:


data1.tail()


# In[23]:


#Reset the index column
data1.reset_index(drop=True)


# In[24]:


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

# In[26]:


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


# In[27]:


sns.violinplot(data=data1["Solar.R"], color='lightgreen')
plt.title("Violin Plot")


# In[28]:


plt.figure(figure=(6,2))
plt.boxplot(data1["Ozone"], vert=False)


# In[29]:


# Extract outliers from boxplot for Ozone column
plt.figure(figure=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']] 


# #### Method 2 for outlier detection
# - Using mu+/-3*sigma limits(Standard deciation method)

# In[31]:


data1["Ozone"].describe()


# In[32]:


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

# In[35]:


import scipy.stats as stats

#Create Q-Q plot
plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q Plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# ## Observation from Q-Q plot 
# - The data does not follow normal distribution as the data points are deviating significantly away from the red line.
# 
# - The data shows a right-skewed distribution and possible outliers

# #### Other visualisations that could help understand the data

# In[38]:


# Create a figure for violin plot

sns.violinplot(data=data1["Ozone"], color='lavender')
plt.title("Violin Plot")

#Show the plot
plt.show()


# In[39]:


sns.swarmplot(data=data1, x = "Weather", y = "Ozone", color="orange",palette="Set2", size=6)


# In[40]:


sns.stripplot(data=data1, x = "Weather", y = "Ozone",color="orange", palette="Set1", size=6, jitter = True)


# In[41]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="blue")
sns.rugplot(data=data1["Ozone"],color="black")


# In[42]:


# Category wise boxplot for ozone
sns.boxplot(data = data1, x = "Weather", y="Ozone")


# ### Correlation coefficient and pair plots

# In[44]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[45]:


# Compute person correlation coefficient
# Between wind speed and temperature
data1["Wind"].corr(data1["Temp"])


# ### Observation
# The correlation between wind and temp is observed to be negatively correlated with mild strength

# In[86]:


data.info()


# In[88]:


# Read all numeric (continuous) columns into a new table data1_numeric 
data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[94]:


# print correlation coefficients for all the above columns
data1_numeric.corr()


# ### Observation
# - The highest correlation strength is observed between Ozone and Temperature (0.597087)
# - The next highest cprrelation strength is observed between Ozone and wind (-0.523738)
# - The next highest correlation strength is observed between wind and Temp (-0.441247)
# - The last correlation strength is observed between Solar and wind (0.257369)

# In[98]:


# Plot a pair plot between all numeric columns using seaborn 
sns.pairplot(data1_numeric)


# In[102]:


# Creating dummy variable for Weather column
data2=pd.get_dummies(data1, columns=['Month','Weather'])
data2


# In[104]:


data1_numeric.values


# In[123]:


data1_numeric


# In[127]:


# Normalization of the data
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

array = data1_numeric.values
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(array)

#transformed data
set_printoptions(precision=2)
print(rescaledX[0:10,:])


# In[133]:


# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler

array = data1_numeric.values
scaler = StandardScaler()
rescaledX = scaler.fit_transform(array)

# Summarize transformed data
set_printoptions(precision=2)
print(rescaledX[0:10,:])


# In[ ]:




