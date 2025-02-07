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

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[3]:


# Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[9]:


# Rearrange the columns
cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# ### Description of columns
# - MPG: Milege of the car (Mile per Gallon) (This is Y-column to be predicted)
# - HP: Horse Power of the car (X! column)
# - VOL: Volume of the car (size) (X2 column)
# - SP: Top speed of the car(Miles per Hour)(X3 column)
# - WT: Weight of the car(Pounds) (X4 column) 

# ## EDA

# In[13]:


cars.info()


# In[15]:


# check for missing values
cars.isna().sum()


# ## Observations
# - There are no missing values
# - There are 81 observations (81 different cars data)
# - the data types of the columns are also relevant and valid

# In[18]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='') # Remove x Label for the boxplot

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust Layout
plt.tight_layout()
plt.show()


# ### Observations from boxplot and histograms
# - There are some extreme values (outliers) observed in towards the right tail of SP and HP distributions.
# - In VOL and WT columns, a few outliers are observed in both tails of thier distributions.
# - The extreme values of cars data may have come from the specify designed nature of cars
# - As this is multi-dimensional data, the outliers with respect to spatial dimensions may have to be considered while building the regression mode

# ## Checking for duplicated rows

# In[22]:


cars[cars.duplicated()]


# ## Pair plots and correlation Coefficients

# In[25]:


# Pairs plot
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[26]:


cars.corr()


# ### Observations from correlation plots and Coefficients
# - The highset correlation strength is observed b/w Weight and Volume
# - The next higher correlation strength is observed b/w HP and SP
# - The next higher correlation strength is observed b/w SP and MPG
# - Between x and y, all the x variables are showing moderate to high correaltion highest being b/w HP and MPG
# - Therefore this dataset qualifiers for buliding a multiple linear regression model to predict MPG
# - Among x columns (x1,x2,x3andx4), some very high correlation strengths are observerd b/w SP vs HP, VOL vs WT
# - The high correlation among x columns is not desirable as it might lead to multicollineraity prob

# # Preparing a preliminary model considering all X columns

# In[29]:


#Build model
#import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[30]:


model1.summary()


# ## Observations from model summary
# - The R-squared and adjusted values are good and about 75% of variability in Y is explanined by X columns
# - The probability value with respect to F-statistic is close to zero, including that all or some of X columns are significant
# - The p-value for VOL and WT are higher than 5% including some interaction issue among themselves, which need to be further explored

# ## Performance metrics for model1

# In[35]:


## Find the performance metrics
## Create a data frame with actual y and predicted y columns

df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[36]:


# Predict for the given X data columns

pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[37]:


# Compute the Mean Squared Error for model1
from sklearn.metrics import mean_squared_error
print("MSE :", mean_squared_error(df1["actual_y1"], df1["pred_y1"]))


# In[45]:


# Compute the Mean Squared error (MSE), RMSE for model1

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# In[47]:


cars.head()


# In[49]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# ## Observations
# - The ideal range of VIF values shall be between 0 to 10.However slightly higher values can be tolerated
# - As seen from the very high VIF values for VOL and WT, it is clear that they are prone multicollinearity problem.
# - Hence it is decided to drop one of the columns (either VOl or WT) to overcomethe multicollinearity.
# - It is decided to drop WT and retain VOL column in further models.

# In[76]:


## Build model
# import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars).fit()


# In[78]:


model2.summary()


# ## Performance metrics for model2

# In[83]:


# Find the Performance metrics
# Create a data frame with actual y and predicted y columns
df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[89]:


# Predict for the given X data columns

pred_y2 = model2.predict(cars.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[93]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# ## Observation
# - The adjused R-squared value improved slightly to 0.76
# - All the p-values for model parameters are less tha 5% hence they are significant
# - Therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPG response variable.
# - There is no improvement in MSE value.

# In[ ]:




