#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import modules
import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('gs://my-bigdata-project-dk/landing/NF-UQ-NIDS-v2.csv',chunksize=1000000)


# In[3]:


df = pd.DataFrame(df.get_chunk(200000))


# In[4]:


# Remove rows where any column has a null value just in case but my data set didnt have any
df = df.dropna()


# In[5]:


# Renaming columns to remove spaces and make names more consistent
df.columns = df.columns.str.replace(' ', '_').str.upper()


# In[6]:


#dropping duplicates
df = df.drop_duplicates()


# In[7]:


df.dtypes


# In[9]:


#dropping unecessary columns
df.drop(['SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES','FTP_COMMAND_RET_CODE','ICMP_IPV4_TYPE','IPV4_SRC_ADDR','IPV4_DST_ADDR'], axis=1, inplace=True)


# In[10]:


df['L7_PROTO'] = df['L7_PROTO'].astype(int)


# In[11]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

new_col= encoder.fit_transform(df['DATASET'])
df['DATASET'] = pd.Series(new_col)

new_col= encoder.fit_transform(df['ATTACK'])
df['ATTACK'] = pd.Series(new_col)


# In[12]:


df.dtypes


# In[13]:


output_path = 'gs://my-bigdata-project-dk/cleaned/data_cleaning.parquet'
df.to_parquet(output_path, index=False)


# In[ ]:




