#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# In[2]:


data = pd.read_csv('iris.data' , header = None)
data


# # Data Pre-processing

# In[16]:


data.columns =(['s1','sw','p1','pw','class'])  # Add column names 
data


# In[17]:


data.shape


# In[24]:


data.info()


# In[25]:


data.describe()


# In[26]:


data.min()


# In[27]:


data.max()


# In[30]:


data.duplicated().sum()   # Is there duplicate values or not and sum it


# In[32]:


data.loc[data.duplicated(), :]    # It keeps 1st record of each categry


# In[33]:


data.describe(include = 'all')


# In[34]:


data.describe(include = 'object')


# In[36]:


data.groupby('s1').groups


# In[37]:


data.groupby('sw').groups


# In[39]:


data.groupby('p1').groups


# In[41]:


data.groupby('pw').groups


# In[38]:


data.groupby('class').groups


# # Visualization

# In[71]:


g = sns.pairplot(data, hue='class', markers='*')
plt.show()


# # Training and Testing the data

# In[42]:


X = data.drop(['class'], axis=1)
y = data['class']
# print(X.head())
print(X.shape)
# print(y.head())
print(y.shape)


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# # KNN Algirithm

# In[49]:


k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))


# In[53]:


plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of KNN')
plt.show()


# In[54]:


log_reg = LogisticRegression()
log_reg.fit(X, y)
y_pred = log_reg.predict(X)
print(metrics.accuracy_score(y, y_pred))


# In[58]:


knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X, y)


# In[63]:


knn.predict([[1.0, 0.5, 0.2, 4.1]])


# In[ ]:




