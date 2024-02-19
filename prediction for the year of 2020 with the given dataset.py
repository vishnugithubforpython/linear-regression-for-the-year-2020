#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[11]:


from sklearn import linear_model


# In[12]:


df=pd.read_csv("C://Users//mvish//Desktop//canada_per_capita_income.csv")
df


# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(df["year"],df["per capita income (US$)"],marker="+",color="red")
plt.xlabel("year")
plt.ylabel("per capita income(US$)")


# In[43]:


reg=linear_model.LinearRegression()
X = df["year"].values.reshape(-1, 1)
y = df["per capita income (US$)"].values
reg.fit(X, y)


# In[46]:


reg.predict([[2020]])


# In[49]:


reg.coef_


# In[50]:


reg.intercept_


# In[55]:


828.46507522*2020+(-1632210.7578554575)


# In[ ]:




