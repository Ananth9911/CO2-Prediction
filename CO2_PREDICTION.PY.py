#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[69]:


df=pd.read_csv('FuelConsumptionCo2.csv')


# In[70]:


df.shape


# In[71]:


df.head()


# In[72]:


k=df[['VEHICLECLASS','ENGINESIZE','CO2EMISSIONS']]
k.head()


# In[73]:


sns.set()
sns.scatterplot('ENGINESIZE','CO2EMISSIONS',data=k,)


# In[80]:


msk=np.random.rand(len(df)) < 0.8
train=df[msk]
test=df[~msk]


# In[75]:


from sklearn import linear_model
regr=linear_model.LinearRegression()
train_x=np.asanyarray(train[['ENGINESIZE']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
print('Coefficients ',regr.coef_)
print('Intercept ',regr.intercept_)


# In[108]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
regr.predict([[6.8]])


# In[100]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


# In[63]:


from sklearn import linear_model
regr=linear_model.LinearRegression().

train_x=np.asanyarray(train[['ENGINESIZE','FUELCONSUMPTION_COMB','CYLINDERS']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x,train_y)
regr.predict([[1.1,3.6,1]])


# In[58]:


#EVALUATION
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


# In[86]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
poly = PolynomialFeatures(degree=2)
poly1=poly.fit_transform(train_x)
df=linear_model.LinearRegression()
k=df.fit(poly1,train_y)
print ('Coefficients: ', df.coef_)
print ('Intercept: ',df.intercept_)


# In[89]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = df.intercept_[0]+ df.coef_[0][1]*XX+ df.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")


# In[91]:


from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = df.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )


# In[95]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
gh=PolynomialFeatures(degree=3)
poly2=gh.fit_transform(train_x)
q=linear_model.LinearRegression()
f=q.fit(poly2,train_y)
print("coeffiecent ", q.coef_)
print("intercept ",q.intercept_)


# In[99]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = q.intercept_[0]+ q.coef_[0][1]*XX + q.coef_[0][2]*np.power(XX, 2) + q.coef_[0][3]*np.power(XX, 3)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
test_x_poly3 = gh.fit_transform(test_x)
test_y3_ = q.predict(test_x_poly3)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y3_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y3_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y3_ , test_y) )


# In[ ]:




