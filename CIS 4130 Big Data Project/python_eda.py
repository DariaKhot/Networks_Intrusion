#!/usr/bin/env python
# coding: utf-8

# In[59]:


# Import the Modules and set up matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Import modules
import pandas as pd
import numpy as np
# Set Pandas options to always display floats with a decimal point
# (not scientific notation)
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.width', 1000)


# In[60]:


df = pd.read_csv('gs://my-bigdata-project-dk/landing/NF-UQ-NIDS-v2.csv',chunksize=1000000)


# In[61]:


df = pd.DataFrame(df.get_chunk(200000))


# In[62]:


#understand the data frame columns
print(df.info())


# In[63]:


#min/max/avg/stdev for all numeric variables
df.describe()


# In[64]:


#number of null varibales
df.isnull().sum()


# In[65]:


#counting all observations
df.count()


# In[66]:


#number of duplicate observations
df.duplicated().sum()


# In[67]:


#data type of each column
df.dtypes


# In[68]:


#list of variables
df.columns


# In[69]:


#amount of unique variables per column, this will be useful for feature engineering and ensure columns are correct
df.nunique()


# In[70]:


#number of observations and columns
df.shape


# In[71]:


#need to do feature engineering for this column later on since it will be important
plt.figure(figsize=(20, 16)) 
plt.xticks(rotation=45)
plt.hist(df['Attack'],bins=20)
plt.show()


# In[72]:


plt.figure(figsize=(20, 16)) 
plt.xticks(rotation=45)
plt.hist(df['Dataset'],bins=4)
plt.show()


# In[73]:


#Create graphs / charts to show the distribution of data (e.g., histograms for categorical variables).
top_5_ips = df['IPV4_SRC_ADDR'].value_counts().head(5).index.tolist()
filtered_df = df[df['IPV4_SRC_ADDR'].isin(top_5_ips)]

plt.figure(figsize=(20, 16))
plt.xticks(rotation=45)
plt.hist(filtered_df['IPV4_SRC_ADDR'])
plt.show()


# In[74]:


top_5_ips = df['IPV4_DST_ADDR'].value_counts().head(5).index.tolist()
filtered_df = df[df['IPV4_DST_ADDR'].isin(top_5_ips)]

plt.figure(figsize=(20, 16))
plt.xticks(rotation=45)
plt.hist(filtered_df['IPV4_DST_ADDR'])
plt.show()


# In[75]:


#checking for class imbalance
plt.figure(figsize=(10, 6)) 
plt.hist(df['Label'],bins=2)
plt.show()


# In[76]:


numeric_columns = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'IN_PKTS', 
                   'OUT_BYTES', 'OUT_PKTS', 'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 
                   'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT', 'MIN_TTL', 'MAX_TTL', 
                   'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN', 
                   'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES', 'RETRANSMITTED_IN_BYTES', 
                   'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS', 
                   'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT', 'NUM_PKTS_UP_TO_128_BYTES', 
                   'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES', 
                   'NUM_PKTS_1024_TO_1514_BYTES', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'ICMP_TYPE', 
                   'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER', 
                   'FTP_COMMAND_RET_CODE', 'Label']

# used heatmap table from another notebook on kaggle bc the one i was trying to run refused to run and crash kernel
df_selected = df[numeric_columns]
plt.figure(figsize = (20,20))
cmap = sns.diverging_palette(500, 10, as_cmap=True)
attrb_reln = sns.heatmap(df_selected.corr(),linewidths=0.5, cmap=cmap)  
plt.savefig('correlations_dataset.jpg', dpi=1000)


# In[ ]:




