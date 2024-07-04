#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
df = pd.read_csv('wine.csv')
df


# In[13]:


from sklearn.datasets import load_wine

wine = load_wine()

wine.keys()

X = wine['data']
y= wine['target']

print(X.shape)
print(y.shape)


# In[14]:


from sklearn.model_selection import train_test_split
    
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)



# In[15]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)



# In[16]:


y_pred = model.predict(X_test)
y_pred


# In[18]:


import pandas as pd 

df = pd.DataFrame(X,columns=wine['feature_names'])

df['target'] = y 
df.sample()


# In[20]:


flower[('target_names')]


# In[21]:


from sklearn.metrics import accuracy_score,\
confusion_matrix,\
classification_report 

cm = confusion_matrix(y_test,y_pred)
cm


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm,annot= True)
plt.show()


# In[23]:


accuracy = accuracy_score(y_test,y_pred)

accuracy


# In[24]:


cr = classification_report(y_test, y_pred)
print(cr)


# In[ ]:




