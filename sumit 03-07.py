#!/usr/bin/env python
# coding: utf-8

# # Answer A

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print('done')


# In[18]:


data=pd.read_csv('heart.csv')


# In[19]:


data


# In[20]:


data.sample()


# In[21]:


data.info()


# In[22]:


data.replace('?',0,inplace=True)


# In[23]:


X = data.drop('target',axis=1)
y = data['target']


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[25]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[26]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

model.fit(X_train,y_train)


# In[27]:


y_pred = model.predict(X_test)
y_pred


# In[28]:


model.score(X_test,y_test)


# In[29]:


accuracy_score(y_test,y_pred)


# In[30]:


cm = confusion_matrix(y_test,y_pred)
cm


# In[31]:


print(classification_report(y_test,y_pred))


# # Answer B

# In[32]:


data.shape


# # Answer C 1

# In[34]:


male_count = (df['sex'] == 1).sum()
female_count = (df['sex'] == 0).sum()
print(f'Number of males: {male_count}')
print(f'Number of females: {female_count}')


# # Answer C 2

# In[42]:


bins = [30,40,50,60,70,80]
labels = ['30-40','40-50','50-60','60-70','70-80']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

age_group = df['age_group'].value_counts().sort_index()
print("total patients of each age group",age_group)

common = age_group.idxmax()
print("most common age group is: ",common)


# # Answer D

# In[37]:


df = pd.read_csv('heart.csv')
columns_of_interest = ['trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'ca']
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
for i, column in enumerate(columns_of_interest):
    row = i // 3
    col = i % 3
    sns.histplot(df[column], ax=axes[row, col], kde=True)
    axes[row, col].set_title(f'Distribution of {column}')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




