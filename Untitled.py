#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt


# In[5]:


import pandas as pd
other_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv


# In[6]:


df = pd.read_csv(other_path, header=None)


# In[7]:


print ('The first 5 rows of the dataframe')
df.head()


# In[10]:


df.tail()


# In[12]:


#Take a look at our dataset; pandas automatically set the header by an integer from 0.

#To better describe our data we can introduce a header, this information is available at: https://archive.ics.uci.edu/ml/datasets/Automobile

#Thus, we have to add headers manually.

#Firstly, we create a list "headers" that include all column names in order. Then, we use dataframe.columns = headers to replace the headers by the list we created.


# In[15]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


# In[18]:


df.columns = headers
df.head(10)


# In[19]:


df1=df.replace('?',np.NaN)


# In[20]:


df=df1.dropna(subset = ["price"], axis=0)
df.head()


# In[21]:


df.to_csv("automobile.csv", index=False)


# In[22]:


df.dtypes


# In[30]:


df.describe()


# In[32]:


df.describe(include = "all")


# In[33]:





# In[34]:


df.replace("?", np.nan, inplace = True)
df.head()


# In[36]:


missing_data = df.isnull()


# missing_data.head()

# In[37]:


missing_data.head(5)


# In[38]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print ("")


# In[39]:


avg_norm_loss = df["normalized-losses"].astype("float").mean(axis= 0)
print("Average of normalized-losses:", avg_norm_loss)


# In[40]:


df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace = True)


# In[41]:


avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)


# In[44]:


df["bore"].replace(np.nan, avg_bore, inplace = True)


# In[45]:


avg_stroke = df["stroke"].astype("float").mean(axis=0)


# In[46]:


df["stroke"].replace(np.nan, avg_stroke, inplace = True)


# In[47]:


avg_horsepower = df["horsepower"].astype("float").mean(axis=0)
print ("Average of horsepower:", avg_horsepower)


# In[48]:


df["horsepower"].replace(np.nan, avg_horsepower, inplace = True)


# In[54]:


avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)


# In[55]:


df["peak-rpm"].replace(np.nan, avg_peakrpm, inplace = True)


# In[57]:


df['num-of-doors'].value_counts()


# In[62]:


df['num-of-doors'].value_counts().idxmax()


# In[63]:


df['num-of-doors'].replace(np.nan, "four", inplace = True)


# In[64]:


df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop= True, inplace=True)


# In[65]:


df.head()


# In[69]:


df.dtypes


# In[72]:


df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")


# In[75]:


df.dtypes


# In[76]:


df['city-L/100km'] = 235/df["city-mpg"]


# In[77]:


df.head()


# In[79]:


df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()


# In[80]:


df['height'] = df['height']/df['height'].max() 
df[['length', 'width', 'height']].head()


# In[84]:


df["horsepower"]=df["horsepower"].astype(int, copy=True)


# In[85]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[86]:


bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


# In[87]:


group_names = ['Low', 'Medium', 'High']


# In[88]:


df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)


# In[89]:


df["horsepower-binned"].value_counts()


# In[90]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[91]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot


# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[92]:


df.columns


# In[93]:


dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()


# In[94]:


dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()


# In[95]:


df = pd.concat([df, dummy_variable_1], axis=1)


# In[96]:


df.drop("fuel-type", axis = 1, inplace=True)


# In[97]:


df.head()


# In[98]:


dummy_variable_2 = pd.get_dummies(df["aspiration"])
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo':'aspiration-turbo'}, inplace=True)
dummy_variable_2.head()


# In[99]:


df = pd.concat([df, dummy_variable_2], axis=1)
df.drop('aspiration', axis=1, inplace=True)


# In[ ]:




