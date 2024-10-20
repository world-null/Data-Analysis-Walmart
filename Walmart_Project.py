#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


# In[3]:


train_df=pd.read_csv("E:/CODING/Data Science/Numpy_pandas_/walmart Data Analysis-20230204T170209Z-001/walmart Data Analysis/train.csv")
test_df=pd.read_csv("E:/CODING/Data Science/Numpy_pandas_/walmart Data Analysis-20230204T170209Z-001/walmart Data Analysis/test.csv")
features_df=pd.read_csv("E:/CODING/Data Science/Numpy_pandas_/walmart Data Analysis-20230204T170209Z-001/walmart Data Analysis/features.csv")
stores_df=pd.read_csv("E:/CODING/Data Science/Numpy_pandas_/walmart Data Analysis-20230204T170209Z-001/walmart Data Analysis/stores.csv")


# In[4]:


train_df


# In[5]:


train_df.isna().sum()


# In[6]:


(train_df["Weekly_Sales"]<0).value_counts()


# In[9]:


negsalesindex=train_df[train_df["Weekly_Sales"]<0].index
train_df.drop(negsalesindex)


# In[13]:


test_df.head()


# In[14]:


test_df.columns


# In[15]:


test_df.describe()


# In[16]:


test_df.isnull().sum()


# In[17]:


features_df.head()


# In[18]:


features_df.info()


# In[19]:


features_df.describe()


# In[20]:


features_df.isnull().sum()


# In[21]:


features_df.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis = 1)


# In[22]:


# (fill_method = mean, mode, median, constant_value, forwardfill, backwardfill, linear)
features_df["CPI"].fillna(features_df["CPI"].mean())


# In[23]:


stores_df.head()


# In[25]:


stores_df.describe()


# In[26]:


stores_df.isnull().sum()


# In[27]:


dropped_stores = stores_df.drop(44)
dropped_stores


# In[28]:


dropped_features = features_df.drop(features_df[features_df["Store"]==1].index)
dropped_features


# types of merging
# inner, outer, left, right

# In[29]:


stores_df


# In[31]:


dataset_m = features_df.merge(stores_df, how= 'inner', on = 'Store')
dataset_m.head()


# In[32]:


dataset_m


# In[33]:


dataset_m.info()


# In[34]:


dataset_m.isnull().sum()


# In[36]:


today = pd.to_datetime('2024-10-20')
print(today)

print(today.day)
print(today.week)
print(today.month)
print(today.year)
print(today.weekday())


# In[37]:


pd.to_datetime(dataset_m["Date"]).dt.week


# In[38]:


dataset_m["Date"] = pd.to_datetime(dataset_m["Date"])


# In[39]:


dataset_m["day"] = dataset_m["Date"].dt.day
dataset_m["week"] = dataset_m["Date"].dt.week
dataset_m["month"] = dataset_m["Date"].dt.month
dataset_m["weekday"] = dataset_m["Date"].dt.weekday

dataset_m


# In[69]:


dataset_m["week"]


# In[40]:


dataset_m["weekday"].value_counts()


# In[41]:


dataset_m.tail()


# In[44]:


train_df[train_df["Store"]==1]


# In[43]:


train_df[(train_df["Store"]==1) & (train_df["Dept"]==1)]


# In[45]:


train_df[(train_df["Store"]==1) & (train_df["Dept"]==2)]


# In[46]:


features_df[features_df["Store"]==1]


# In[73]:


train_df_1 = train_df.merge(features_df, how = "inner", on= ["Store", "Date"])
train_df_1


# In[48]:


train_df_1= train_df_1.sort_values(["Store", "Dept", "Date"])


# In[71]:


data


# In[ ]:





# VISUALIZATIONS

# 1) scatter plot

# In[49]:


#scatter plot bw Weekly slaes vs Store
plt.figure(figsize = [15,8])
sns.scatterplot(x = train_df_1["Store"], y = train_df_1["Weekly_Sales"])


# In[50]:


df = train_df[train_df_1["Store"]==6]
sns.histplot(x = df["Weekly_Sales"])


# In[51]:


plt.figure(figsize = [30, 18])

for i in range(1, 46):
    plt.subplot(5, 9, i)

    df = train_df[train_df_1["Store"]==i]
    sns.histplot(x = df["Weekly_Sales"])


# In[52]:


plt.figure(figsize = [12, 6])
ax = sns.barplot(x=train_df_1["Store"], y = train_df_1["Weekly_Sales"])#, estimator = sum)
ax.set_xticklabels(labels = ["Store" + str(i) for i in range(1,46)])
plt.xticks(rotation = 90)
plt.plot()


# In[53]:


myticks = []
for i in range(1, 46, 1):
    myticks.append("Store" + str(i))
myticks
["Store"+str(i) for i in range(1, 46, 1)]

plt.figure(figsize = [12, 6])
sns.barplot(x = train_df_1["Store"], y = np.array(train_df_1["Weekly_Sales"]), palette = "Set2")

locs, ticks = plt.xticks()
plt.xticks(locs, myticks, fontsize = 12, rotation = 90)


# In[54]:


myticks = []
for i in range(1, 46, 1):
    myticks.append("Store" + str(i))
myticks

plt.figure(figsize = [12, 6])
sns.barplot(x = train_df_1["Store"], y = np.array(train_df_1["Weekly_Sales"]), palette = "Set2")

locs, ticks = plt.xticks()
plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 44], ['tick1', 'tick2', 'tick3', 'tick4', 'tick5', 'tick6', 'tick7', 'tick8', 'tick9', 'tick10'], fontsize = 12, rotation = 90)


# In[55]:


np.array(locs)[np.array(locs)%5==0]
np.array(myticks)[np.array(locs)%5==0]


# In[56]:


myticks = ["Store1", "Store2", "Store3", "Store4", "Store5", "Store6", "Store7", "Store8", "Store9", "Store"]


# In[57]:


locs


# In[58]:


plt.figure(figsize = [12, 6])
sns.boxplot(x = train_df_1["Store"], y = train_df_1["Weekly_Sales"])


# In[59]:


plt.figure(figsize = [10, 5])
sns.barplot(x = train_df_1["Dept"], y = train_df_1["Weekly_Sales"])
# plt.xticks(fontsize = 16)
# plt.yticks(fontsize = 16)
plt.xlabel('Department',  fontsize = 20)
plt.ylabel('Weekly Sales', fontsize = 20)
plt.title("Write Title Here")


# In[60]:


train_df_1["Dept"].unique()


# In[61]:


def scatter(train_df_1, column):
    #plot the figure
    plt.figure()
    #plot the scatter plot with data from the specified column in x axis and weekly sales in y axis
    plt.scatter(train_df[column], train_df['Weekly_Sales'])
    #give y label as weekly_sales
    plt.ylabel('Weekly_Sales')
    #Give the xlabel as the column specified as parameter in the function
    plt.xlabel(column)


# In[62]:


#plot a scatter plot using the scatter function a scatter plot of weekly sales with respect to Store
scatter(train_df_1, 'Store')
#plot a scatter plot using the scatter function a scatter plot of weekly sales with respect to Department
scatter(train_df_1, 'Dept')


# 

# In[ ]:




