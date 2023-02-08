#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as mp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[6]:


data= pd.read_csv (C:/Users/user/OneDrive/Documents/Machine Learning/insurance (1).csv)


# In[20]:


data.head()


# In[21]:


data.tail()


# In[22]:


data.describe()


# In[23]:


data.shape


# In[24]:


data.nunique()


# In[25]:


data ['region'].unique()


# In[26]:


data.columns


# In[27]:


data ['children'].unique()


# In[28]:


data ['sex'].unique()


# In[29]:


#cleaning the data.


# In[30]:


data.isnull().sum()


# In[31]:


#histogram
sns.displot(data['charges']);


# In[32]:


sns.relplot(x='age', y='charges', hue='region',data=data)
mp.show()


# In[33]:


corelation = data.corr()


# In[34]:


print(data.corr())
  
# plotting correlation heatmap
dataplot = sns.heatmap(data.corr(), cmap="YlGnBu", annot=True)
  
# displaying heatmap
mp.show()


# In[35]:


sns.pairplot(data)
mp.show()


# In[36]:


sns.relplot(x='age', y='charges', hue='sex',data=data)
mp.show()


# In[37]:


sns.relplot(x='age', y='charges', hue='region',data=data)
mp.show()


# In[38]:


sns.displot(data['bmi'])
mp.show()


# In[39]:


sns.displot(data['region'])
mp.show()


# In[40]:


sns.catplot(x='charges', kind= 'box' , data= data )
mp.show()


# In[41]:


sns.catplot(x='bmi', kind= 'box' , data= data )
mp.show()


# In[42]:


x =  data[['age', 'children']]


# In[43]:


y = data['charges']


# In[44]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[45]:


model = LinearRegression()


# In[46]:


model.fit(x_train, y_train)


# In[47]:


print(model.coef_)


# In[48]:


print(model.intercept_)


# In[49]:


pd.DataFrame(model.coef_, x.columns, columns = ['Coeff'])


# In[50]:


#making predictions from our model


# In[51]:


predictions = model.predict(x_test)


# In[52]:


mp.scatter(y_test, predictions)
mp.show()


# In[53]:


mp.hist(y_test - predictions)
mp.show()


# In[54]:


#testing performance of our model


# In[55]:


metrics.mean_absolute_error(y_test, predictions)


# In[56]:


metrics.mean_squared_error(y_test, predictions)


# In[ ]:




