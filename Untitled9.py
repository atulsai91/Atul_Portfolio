#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv"
df = pd.read_csv(path)
df


# In[3]:


get_ipython().run_cell_magic('capture', '', '! pip install seaborn')


# In[4]:


import matplotlib as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


print(df.dtypes)


# In[7]:


df.corr()


# In[8]:


df[['bore', 'stroke', 'compression-ratio','horsepower']].corr()


# In[13]:


sns.regplot(x="engine-size",y="price", data=df)
plt.ylim(0,)


# In[10]:


#the engine-size goes up, the price goes up: this indicates a positive direct correlation between these two variables. Engine size seems like a pretty good predictor of price since the regression line is almost a perfect diagonal line.


# In[11]:


df[['engine-size', 'price']].corr()


# In[12]:


#the correlation between 'engine-size' and 'price' and see it's approximately 0.87


# In[14]:


sns.regplot(x="highway-mpg", y="price", data= df)


# In[15]:


#the highway-mpg goes up, the price goes down: this indicates an inverse/negative relationship between these two variables. Highway mpg could potentially be a predictor of price.


# In[16]:


df[["highway-mpg", "price"]].corr()


# In[17]:


#can examine the correlation between 'highway-mpg' and 'price' and see it's approximately -0.704


# In[18]:


sns.regplot(x="peak-rpm", y="price", data=df)


# In[19]:


#Peak rpm does not seem like a good predictor of the price at all since the regression line is close to horizontal. Also, the data points are very scattered and far from the fitted line, showing lots of variability. 


# In[20]:


df[['peak-rpm','price']].corr()


# In[21]:


sns.regplot(x="stroke", y="price", data = df)


# In[22]:


df[["stroke", "price"]].corr()


# In[26]:


sns.regplot(x="stroke", y= "price", data = df)


# In[28]:


sns.boxplot(x="body-style", y="price", data=df)


# In[29]:


#We see that the distributions of price between the different body-style categories have a significant overlap, and so body-style would not be a good predictor of price. Let's examine engine "engine-location" and "price":


# In[30]:


sns.boxplot(x="engine-location", y="price", data=df)


# In[31]:


#we see that the distribution of price between these two engine-location categories, front and rear, are distinct enough to take engine-location as a potential good predictor of price.


# In[32]:


sns.boxplot(x="drive-wheels", y="price", data=df)


# In[33]:


# we see that the distribution of price between the different drive-wheels categories differs; as such drive-wheels could potentially be a predictor of price.


# In[34]:


df.describe()


# In[35]:


df.describe(include=['object'])


# In[36]:


df['drive-wheels'].value_counts()


# In[38]:


engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)


# In[39]:


#This is because we only have three cars with a rear engine and 198 with an engine in the front, this result is skewed. Thus, we are not able to draw any conclusions about the engine location.


# In[40]:


# The data is grouped based on one or several variables and analysis is performed on the individual groups.


# In[41]:


df['drive-wheels'].unique()


# In[42]:


df_group_one = df[['drive-wheels','body-style','price']]


# In[43]:


df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one


# In[44]:


#You can also group with multiple variables. For example, let's group by both 'drive-wheels' and 'body-style'. This groups the dataframe by the unique combinations 'drive-wheels' and 'body-style'. We can store the results in the variable 'grouped_test1'


# In[45]:


df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1


# In[46]:


grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot


# In[47]:


grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot


# In[48]:


df_gptest2 = df[['body-style','price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'],as_index= False).mean()
grouped_test_bodystyle


# In[49]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()


# In[51]:


fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[52]:


df.corr()


# In[53]:


#using "stats" module in the "scipy" library.


# In[54]:


from scipy import stats


# In[55]:


pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In[56]:


#Since the p-value is  <  0.001, the correlation between wheel-base and price is statistically significant, although the linear relationship isn't extremely strong (~0.585)


# In[57]:


pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  


# In[58]:


#The Pearson Correlation Coefficient is 0.8344145257702843  with a P-value of P =  2.189577238894065e-53


# In[59]:


pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# In[60]:


#Since the p-value is  <  0.001, the correlation between engine-size and price is statistically significant, and the linear relationship is very strong (~0.872).


# In[61]:


pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value ) 


# In[62]:


#Since the p-value is  <  0.001, the correlation between bore and price is statistically significant, but the linear relationship is only moderate (~0.521)


# In[63]:


pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value) 


# In[64]:


#Since the p-value is  <  0.001, the correlation between city-mpg and price is statistically significant, and the coefficient of ~ -0.687 shows that the relationship is negative and moderately strong.


# In[65]:


pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value ) 


# In[66]:


#Since the p-value is < 0.001, the correlation between highway-mpg and price is statistically significant, and the coefficient of ~ -0.705 shows that the relationship is negative and moderately strong.


# In[67]:


#ANOVA analyzes the difference between different groups of the same variable, the groupby function will come in handy. Because the ANOVA algorithm averages the data automatically, we do not need to take the average before hand.


# In[69]:


grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)


# In[70]:


df_gptest


# In[71]:


grouped_test2.get_group('4wd')['price']


# In[72]:


# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)   


# In[73]:


f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )


# In[74]:


f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val)


# In[75]:


f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  
 
print("ANOVA results: F=", f_val, ", P =", p_val) 


# In[ ]:




