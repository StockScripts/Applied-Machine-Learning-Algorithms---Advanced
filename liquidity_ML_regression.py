#!/usr/bin/env python
# coding: utf-8

# In[1]:


# SciKit-learn package

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
liquid = pd.read_csv('liquidity_data.csv')


# In[2]:


liquid.head(3)


# In[3]:


liquid.describe()


# In[4]:


target = liquid.available_liquidity
inputs = liquid.drop('available_liquidity', axis=1)


# In[5]:


results = train_test_split(inputs, target, test_size = 0.2, random_state = 1)


# In[6]:


print(type(results))
print(len(results))
print('---')
for item in results:
    print(item.shape)


# In[7]:


input_train, input_test, target_train, target_test = results


# In[8]:


from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipelines = {
    'lasso' : make_pipeline(StandardScaler(), Lasso(random_state=1)),
    'ridge' : make_pipeline(StandardScaler(), Ridge(random_state=1))
}


# In[10]:


# disct_name['key name'] = value

from sklearn.linear_model import ElasticNet

pipelines['enet'] = make_pipeline(StandardScaler(), ElasticNet(random_state=1))


# In[15]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

pipelines['rf'] = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=1))
pipelines['gb'] = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=1))


# In[20]:


lasso_hyperparameters = {
    'lasso__alpha' : [0.01, 0.05, 0.1, 0.5, 1, 5]
}
ridge_hyperrparameters = {
    'ridge__alpha' : [0.01, 0.05, 0.1, 0.5, 1, 5]
}
enet_hyperparameters = {
    'elasticnet__alpha' : [0.01, 0.05, 0.1, 0.5, 1, 5], 
    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]
}


# In[21]:


rf_hyperparameters = {
    'randomforestregressor__n_estimators' : [100, 200],
    'randomforestregressor__max_features' : ['auto', 0.3, 0.6]
}
gb_hyperparameters = {
    'gradientboostingregressor__n_estimators' : [100, 200],
    'gradientboostingregressor__learning_rate' : [0.05, 0.1, 0.2],
    'gradientboostingregressor__max_depth' : [1, 3, 5]
}


# In[23]:


hyperparameter_grids = {
    'lasso' : lasso_hyperparameters,
    'ridge' : ridge_hyperrparameters,
    'enet' : enet_hyperparameters,
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters
}


# In[25]:


for key in ['enet', 'gb', 'ridge', 'rf', 'lasso']:
    if key in hyperparameter_grids:
        if type(hyperparameter_grids[key]) is dict:
            print( key, 'was found, and it is a grid')
        else:
            print(key, 'was found, but it is not a grid')
    else:
        print(key, 'was not found')


# In[26]:


from sklearn.model_selection import GridSearchCV

untrained_lasso_model = GridSearchCV(pipelines['lasso'], hyperparameter_grids['lasso'], cv=5)


# In[27]:


print(pipelines.keys())
print(hyperparameter_grids.keys())


# In[28]:


models = {}


# In[29]:


for i in pipelines.keys():
    models[i] = GridSearchCV(pipelines[i], hyperparameter_grids[i], cv=5)
    
models.keys()


# In[30]:


models['lasso'].fit(input_train, target_train)


# In[31]:


for m in models.keys():
    models[m].fit(input_train, target_train)
    print(m, "is trained and tuned")


# In[32]:


from sklearn.metrics import r2_score, mean_absolute_error


# In[33]:


lasso_preds = models['lasso'].predict(input_test)
print('R-Squarred:', round(r2_score(target_test, lasso_preds), 3))
print('MAE:', round(mean_absolute_error(target_test, lasso_preds), 3))


# In[35]:


for mod in models.keys():
    preds = models[mod].predict(input_test)
    print(mod + ':')
    print('R-Squarred:', round(r2_score(target_test, preds), 3))
    print('MAE:', round(mean_absolute_error(target_test, preds), 3))


# In[38]:


# Winning gradiant boosting model

preds = models['gb'].predict(input_test)
plt.scatter(preds, target_test)

plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()


# In[42]:


client = pd.read_csv('client.csv')


# In[44]:


x = models['gb'].predict(client)


# In[45]:


x


# In[ ]:





# In[ ]:




