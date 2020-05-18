#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[12]:


stock_data = pd.read_csv('stock_data_v2.csv')
stock_data


# In[11]:


stock_data.Sector.unique()


# In[15]:


sns.countplot(y='Sector', data=stock_data)
plt.show()


# In[19]:


stock_data.Sector.replace('Information Technology .', 'Information Technology', inplace=True)
stock_data.Sector.replace('Enersy', 'Energy', inplace=True)
stock_data.Sector.replace(['lndustrials', 'Industrial$'], 'Industrials', inplace=True)
sns.countplot(y='Sector', data=stock_data)
plt.show()


# In[20]:


stock_data.Sector.replace('Materials', 'Industrials', inplace=True)
stock_data.Sector.replace(['Telecommunication Services', 'Utilities', 'Real Estate'], 'Other', inplace=True)
sns.countplot(y='Sector', data=stock_data)
plt.show()


# In[30]:


stock_data.shape


# In[34]:


stock_data.hist(figsize=(15,10))
plt.show


# In[35]:


stock_data[stock_data['Current Price'] > 10000]


# In[36]:


stock_data_sans_brk = stock_data[stock_data['Current Price'] < 300000]
stock_data_sans_brk.hist(figsize=(5,7))
plt.show()


# In[37]:


stock_data[stock_data['Current Price'] > 1000]


# In[40]:


stock_data_sans_outlier = stock_data[stock_data['Current Price'] < 1000]
stock_data_sans_outlier.hist(figsize=(5,7))
plt.show()


# In[42]:


stock_data.describe()


# In[47]:


pd.isnull(stock_data['Price Target'])


# In[46]:


stock_data[pd.isnull(stock_data['Price Target'])]
# These are ETFs


# In[49]:


# Drop ETFs
stock_data = stock_data.dropna()
stock_data.describe()


# In[51]:


sns.boxplot(y='Sector', x='Beta', data=stock_data)
plt.show()


# In[52]:


sns.boxplot(y='Sector', x='Return', data=stock_data)
plt.show()


# In[53]:


stock_data.to_csv('stock_data.csv', index=None)


# In[ ]:




