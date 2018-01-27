
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
PATH = "python/Kaggle/House Prices/"  #where you put the files


# In[12]:


#df = pd.read_csv('train.csv', index_col='Id')

df_train = pd.read_csv(PATH + 'train.csv', index_col='Id')
df_test = pd.read_csv(PATH + 'test.csv', index_col='Id')


# In[15]:


#Preparing the tables

target = df_train['SalePrice']  #target variable
df_train = df_train.drop('SalePrice', axis=1)
df_train['training_set'] = True
df_test['training_set'] = False


# In[17]:


#

df_full = pd.concat([df_train, df_test])
df_full = df_full.interpolate()   #check documentation
df_full = pd.get_dummies(df_full)   #check documentation


# In[18]:


#Separating tables again

df_train = df_full[df_full['training_set']==True]
df_train = df_train.drop('training_set', axis=1)

df_test = df_full[df_full['training_set']==False]
df_test = df_test.drop('training_set', axis=1)


# In[19]:


#Training

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(df_train, target)


# In[20]:


#Results

preds = rf.predict(df_test)
my_submission = pd.DataFrame({'Id': df_test.index, 'SalePrice': preds})
my_submission.to_csv(PATH + 'submission.csv', index=False)

