#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Example Case: 
# House Sale Price Prediction (like Zillow's "zestimate") 

# When you see a line starting with "TASK", do that task!

# ### TASK: Click on the next cell and press shift-enter
# You will get the code in it get executed.   
# The result of last command or representation of last varible in that cell will be displayed 

# In[13]:


import pandas as pd
housing = pd.read_csv('data/housing_processed.csv')
housing.head()


# ### Filtering Columns
# Some columns were not removed when equivalent coded ones were created

# In[86]:


housing[["ExterQual","ExterQual_Coded"]].head()


# ### Filtering in a series
# dtypes returns a series   
# filtering series and dataframes are similar

# In[18]:


type(housing.dtypes==object)


# In[19]:


housing.dtypes[housing.dtypes==object]


# In[20]:


housing.dtypes[housing.dtypes==object].shape


# In[21]:


"SalePrice" in housing.columns 


# ### Removing Undesired Columns
# In my case, my colleague had left above non-numeric columns in preprocessing, after creating corresponding coded versions

# In[22]:


len(housing.columns)


# In[93]:


# We could drop columns by name:
housing_ml = housing.drop(columns=["ExterQual"])


# In[24]:


# or wholesale, keeping only numeric:
housing_ml = housing.loc[:,housing.dtypes != object]


# In[25]:


len(housing_ml.columns)


# # Separate Target into new Variable
# - "SalePrice" is the target.    
#  - The value we want to predict from other values (features) for a house.  
# - Currently it is a column like the other features.   
# - Scikit-learn needs 2 variables: features (X) and target (y) to be Predicted into its own 1-D array 

# # NumPy
# - Both Pandas and scikit-learn are build on top of NumPy
# - scikit-learn can not directly work on dataframes
# - X and y data type needs to be NumPy "ndarrays"

# In[26]:


housing_ml.shape


# In[68]:


# Split data as features and target
# take "SalePrice" values into its own 1-D array 
sale_price = housing_ml.pop('SalePrice')
type(sale_price)


# In[72]:


# pop removes the column
# "in place" operation
# now housing_ml has one less column
housing_ml.shape


# In[74]:


y = sale_price.values
type(y)


# # See what other methods are available for ndarray

# In[27]:


# press tab after putting cursor after dot "."
#y. #uncomment, press tab after . 


# In[30]:


y.shape
# (1460,)
# it is equivalent to (1460)
# means it is a 1-d array


# ### TASK: get ndarray version of feature dataframe put it onto variable X

# In[83]:


X = housing_ml.values


# ### TASK: check the shape of X

# In[32]:


X.shape


# ### TASK: programmatically check if X and y has matching number of rows
# You

# In[33]:


X.shape[0] == y.shape[0]


# # First Model
# Q: What would you do if you had no features?

# A: You would always estimatate the average house price.

# We will have to do much better than that.  
# We have so much data to base our decision on.   
# It can still serve us as a baseline to compare.   
# An inferior baseline could be: random in the range or max and min in training data. 

# In[36]:


# Import estimator
import sklearn
from sklearn.dummy import DummyRegressor
# Instantiate estimator
# guess the mean every single time
mean_reg = DummyRegressor(strategy='mean')
# fit estimator
mean_reg.fit(X, y)


# In[38]:


# predict
mean_reg.predict(X)


# ## Evaluating The Model
# scikit-learn regressors have a score function.   
# It gives you how much better your model does compared to worst model
# Technically: what percentage of the variance has decreased over the worst model

# "Mean" *is* the worst model, so its score will be 0.

# In[39]:


mean_reg.score(X, y)


# ## Fitting a linear model 
# First, let's use only one feature 

# In[40]:


from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()


# In[42]:


X_lf = housing_ml[['LotFrontage']]


# In[43]:


linear_model.fit(X_lf, y)


# Above, you see that it used defaults to create the estimator.   
# You could google "LinearRegression sklearn" and find the documentation:
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# to see the options for the other parameters.

# In[44]:


y_pred = linear_model.predict(X_lf)


# In[45]:


linear_model.score(X_lf, y)


# ### Chart Showing the Linear Fit
# matplotlib is the most common visualization library

# In[46]:


# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


# plt.figure(figsize=(12, 5))
# plt.scatter(y, y_pred);


# In[48]:


# plt.scatter(X_lf,y)
# plt.plot(X_lf,y_pred,'r--')


# ### TASK: add labels to these charts
# search label:
# https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py
# 

# ### Task: try replacing scatter with plot
# Do you see why scatter is needed for data rows.
# Try also replacing plot with scatter. 

# # Effect of using a Better predictor 
# Ground Living Area should be better than Lot Frontage!

# In[49]:


X_area = housing_ml[['GrLivArea']]


# In[50]:


linear_model.fit(X_area, y)


# Now the linear_model has another model in it

# In[51]:


y_pred2 = linear_model.predict(X_area)
linear_model.score(X_area, y)


# In[52]:


# plt.figure(figsize=(12, 5))
# plt.scatter(y, y_pred2); # blue obviously better
# plt.scatter(y, y_pred); # orange


# ### TASK: add legend
# which color is the prediction based on which feature

# # Using all predictors!

# In[53]:


# We had 81 columns (80 features) in original dataset,
# coded as 221 features!
X.shape


# In[54]:


linear_model.fit(X, y)


# In[55]:


y_pred3 = linear_model.predict(X)


# In[56]:


linear_model.score(X, y)


# In[57]:


# plt.figure(figsize=(12, 5))
# plt.scatter(y_pred3, y);


# In[58]:


import numpy as np
from sklearn.model_selection import train_test_split
random_state = 21
train_size = .8


# In[ ]:


linear_model = LinearRegression()
X_lf = housing_ml[['YearBuilt']]
linear_model.fit(X_lf, y)
y_pred = linear_model.predict(X_lf)
linear_model.score(X_lf, y)


# In[ ]:

from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

mseDict = {}
scoreDict = {}
for feature in housing_ml.columns:
    linear_model = LinearRegression()
    X_lf = housing_ml[[feature]]
    X_train, X_test, y_train, y_test = train_test_split(X_lf, y, train_size = .8, random_state = 21)
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)
    mseDict[feature] = mean_squared_error(y_test, y_pred, squared = False)
    scoreDict[feature] = linear_model.score(X_test, y_test)

mseDict = sorted(mseDict.items(), key=lambda x: x[1])
scoreDict = sorted(scoreDict.items(), key=lambda x: x[1], reverse = True)

top10 = mseDict[:10]
top10score = scoreDict[:10]
#print(top10score)
print('The best prediction is from', top10score[0][0], 'giving an error score of', top10score[0][1])
print('Top 10 error:')

i = 1
for p in top10:
    print(str(i), ' ', top10[i-1][0], ' ', top10[i-1][1], ', score: ', top10score[i-1][1], sep = '')
    i += 1
    
top10head = []
for predictor in top10:
    top10head.append(predictor[0])

group45 = []
c = 1

for predictor in top10head:
    for index in range(c,10):
        group45.append([predictor, top10head[index]])
    c+=1
    
mseDict2 = {}
scoreDict2 = {}
n = 1

for pair in group45:
    linear_model = LinearRegression()
    X_lf = housing_ml[[pair[0], pair[1]]]
    X_train, X_test, y_train, y_test = train_test_split(X_lf, y, train_size = .8, random_state = 21)
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)
    mseDict2['Model: ' + str(n)] = mean_squared_error(y_test, y_pred, squared = False)
    scoreDict2['Model: ' + str(n)] = linear_model.score(X_test, y_test)
    n += 1

mseDict2 = sorted(mseDict2.items(), key=lambda x: x[1])
scoreDict2 = sorted(scoreDict2.items(), key=lambda x: x[1], reverse = True)

top10_2 = mseDict2[:10]
top10score2 = scoreDict2[:10]
#print(top10score2)
print()
print('The best prediction is from', top10score2[0][0], 'giving an error score of', top10score2[0][1])
print('Top 10 error:')

i = 1
for p in top10_2:
    print(str(i), ' ', top10_2[i-1][0], ' ', top10_2[i-1][1], ', score: ', top10score2[i-1][1], sep = '')
    i += 1

linear_model = LinearRegression()
X_lf = housing_ml
X_train, X_test, y_train, y_test = train_test_split(X_lf, y, train_size = .8, random_state = 21)
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
allMSE = mean_squared_error(y_test, y_pred, squared = False)
allScore = linear_model.score(X_test, y_test)
print()
print('Linear model all features:')
print(allMSE)
print(allScore)

KNR_model = KNeighborsRegressor(n_neighbors=5)
X_lf = housing_ml
X_train, X_test, y_train, y_test = train_test_split(X_lf, y, train_size = .8, random_state = 21)
KNR_model.fit(X_train, y_train)
y_pred = KNR_model.predict(X_test)
knr5MSE = mean_squared_error(y_test, y_pred, squared = False)
knr5Score = KNR_model.score(X_test, y_test)
print()
print('KNR5:')
print(knr5MSE)
print(knr5Score)

KNR_model = KNeighborsRegressor(n_neighbors=10)
X_lf = housing_ml
X_train, X_test, y_train, y_test = train_test_split(X_lf, y, train_size = .8, random_state = 21)
KNR_model.fit(X_train, y_train)
y_pred = KNR_model.predict(X_test)
knr10MSE = mean_squared_error(y_test, y_pred, squared = False)
knr10Score = KNR_model.score(X_test, y_test)
print()
print('KNR10:')
print(knr10MSE)
print(knr10Score)

print('Linear regression, when using all features seems to perform the best')
# In[ ]:




