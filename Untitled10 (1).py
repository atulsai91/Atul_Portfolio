#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[19]:


# path of data 
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


lm = LinearRegression()
lm


# In[22]:


X = df[['highway-mpg']]
Y = df['price']


# In[23]:


lm.fit(X,Y)


# In[24]:


Yhat=lm.predict(X)
Yhat[0:5] 


# In[25]:


lm.intercept_


# In[26]:


lm.coef_


# In[31]:


lm1 = LinearRegression()
lm1


# In[32]:


lm1.fit(df[['engine-size']], df[['price']])
lm1


# In[34]:


lm1.intercept_


# In[35]:


lm1.coef_


# In[36]:


Yhat=-7963.34 + 166.86*X


# In[37]:


Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]


# In[38]:


lm.fit(Z, df['price'])


# In[39]:


lm.intercept_


# In[40]:


lm.coef_


# In[41]:


lm2 = LinearRegression()
lm2.fit(df[['normalized-losses', 'highway-mpg']],df['price'])


# In[42]:


lm2.intercept_
lm2.coef_


# In[43]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[44]:


width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)


# In[45]:


plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)


# In[46]:


df[[ "peak-rpm", "highway-mpg","price"]].corr()


# In[47]:


width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()


# In[48]:


Y_hat = lm.predict(Z)


# In[49]:


plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# In[62]:


def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


# In[63]:


x = df['highway-mpg']
y = df['price']


# In[64]:


#Let's fit the polynomial using the function polyfit, then use the function poly1d to display the polynomial function
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)


# In[53]:


PlotPolly(p, x, y, 'highway-mpg')


# In[65]:


np.polyfit(x, y, 3)
#We can already see from plotting that this polynomial model performs better than the linear model. This is because the generated polynomial function "hits" more of the data points.


# In[55]:


f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p1)
PlotPolly(p1,x,y, 'Highway MPG')


# In[66]:


#We can perform a polynomial transform on multiple features. First, we import the module:
from sklearn.preprocessing import PolynomialFeatures


# In[57]:


pr=PolynomialFeatures(degree=2)
pr


# In[59]:


Z_pr=pr.fit_transform(Z)


# In[60]:


Z.shape


# In[61]:


Z_pr.shape


# In[73]:


#Data Pipelines simplify the steps of processing the data. We use the module Pipeline to create a pipeline. We also use StandardScaler as a step in our pipeline.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[74]:


#PolynomialFeatures object of degree 2:
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]


# In[69]:


#We create the pipeline, by creating a list of tuples including the name of the model or estimator and its corresponding constructor.


# In[75]:


pipe=Pipeline(Input)
pipe


# In[76]:


pipe.fit(Z,y)


# In[77]:


ypipe=pipe.predict(Z)
ypipe[0:4]


# In[78]:


Input=[('scale',StandardScaler()),('model',LinearRegression())]

pipe=Pipeline(Input)

pipe.fit(Z,y)

ypipe=pipe.predict(Z)
ypipe[0:10]


# In[84]:


lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))


# In[85]:


Y_predict_multifit = lm.predict(Z)


# In[86]:


print('The mean square error of price and predicted value using multifit is: ',       mean_squared_error(df['price'], Y_predict_multifit))


# In[ ]:




