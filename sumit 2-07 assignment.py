#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
df = pd.read_csv('iris csv.csv')
df.head()


# # 1

# In[7]:


df.head(8)


# # 2

# In[10]:


odd_numbered_rows = df.iloc[1::2]
shuffled_rows = odd_numbered_rows.sample(frac=1)
print(shuffled_rows)


# # 3

# In[13]:


num_columns = df.shape[1]
print(f"Number of Columns:{num_columns}")
column_names = df.columns
print ("column names:")
for name in column_names:
    print (name)


# # 4

# In[14]:


df.shape


# # 5

# In[15]:


data_sliced = df.iloc[1:50]
new_data = data_sliced
new_data


# In[ ]:





# # 6

# In[21]:


print(df.loc[:,df.columns.str.startswith('petal width')])


# # 7

# In[18]:


df.loc[0]


# # 8

# In[19]:


df.iloc[:,2]


# In[ ]:




