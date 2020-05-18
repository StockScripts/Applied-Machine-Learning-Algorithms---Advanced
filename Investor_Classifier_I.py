#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


investor_data = pd.read_csv('investor_data.csv')
investor_data.head(3)


# In[3]:


investor_data.hist(figsize=(10,7))
plt.show()


# In[4]:


investor_data = investor_data[investor_data.total_fees > 0]
investor_data.hist(figsize=(10,7))
plt.show()


# In[6]:


sns.countplot(y='commit', data=investor_data)
plt.show()


# In[16]:


investor_data.groupby('investor').commit.value_counts().plot(kind='barh')
plt.show()


# In[17]:


investor_data.groupby('invite_tier').commit.value_counts().plot(kind='barh')
plt.show()


# In[18]:


investor_data['tier_change'] = np.where(
    investor_data.prior_tier == investor_data.invite_tier, 'None', np.where(
        investor_data.prior_tier == 'Participant', 'Promoted', 'Demoted'
    )
)
investor_data.head(3)


# In[19]:


investor_data.groupby('tier_change').commit.value_counts().plot(kind='barh')
plt.show()


# In[26]:


investor_data[investor_data['investor'] == 'Goldman Sachs'].groupby('commit').median()


# In[29]:


investor_data['fee_percent'] = investor_data.fee_share / investor_data.total_fees
investor_data['invite_percent'] = investor_data.invite / investor_data.deal_size


# In[32]:


sns.lmplot(x='total_fees', y='fee_percent', hue='commit', data=investor_data, fit_reg=False)
plt.show()

# Shows that issuers are more likely to commit if they are paid larger fees and a greater 
# fee percent... which makes sense


# In[34]:


sns.lmplot(x='deal_size', y='invite_percent', hue='commit', data=investor_data, fit_reg=False)
plt.show()


# In[ ]:




