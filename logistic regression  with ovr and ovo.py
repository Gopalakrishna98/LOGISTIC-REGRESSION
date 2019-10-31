#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt


# In[2]:


get_ipython().system('pip install pandas_profiling')


# create the dataset

# In[3]:


wh=list(np.random.normal(0,10,100));wh.sort()
ww=list(np.random.normal(55,10,100))
cw=[0]*100
mh=list(np.random.normal(28,10,100));mh.sort()
mw=list(np.random.normal(55,10,100))
cm=[1]*100
data=pd.DataFrame([wh+mh,ww+mw,cw+cm]).T
data.columns=["f1","f2","class"]


# In[4]:


data.head()


# In[5]:


ProfileReport(data)


# In[6]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(data.iloc[:,:1],data.iloc[:,2])


# In[7]:


model.coef_


# In[8]:


model.intercept_


# In[9]:


metrics.accuracy_score(data.iloc[:,2],model.predict(data.iloc[:,:1]))


# In[10]:


metrics.accuracy_score(data.iloc[:,2],model.predict(data.iloc[:,1:2]))


# In[11]:


X=data.iloc[:,0:2]


# In[12]:


Y=data[["class"]]


# In[13]:


Y


# In[14]:


model.fit(X,Y)


# In[15]:


Y_pred=model.predict(X)


# In[16]:


metrics.accuracy_score(Y,Y_pred)


# In[17]:


get_ipython().run_line_magic('pinfo2', 'LogisticRegression')


# In[18]:


model.predict_proba(X)


# In[19]:


iris=sns.load_dataset("iris")


# In[20]:


ProfileReport(iris)


# In[41]:


iris.head()


# In[22]:


X=iris.iloc[:,0:4]
#Y=iris.iloc[]
X


# In[23]:


Y=iris.iloc[:,4]
Y


# In[24]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)


# In[25]:


model.fit(X_test,Y_test)


# In[26]:


Y_pred=model.predict(X_test)


# In[27]:


metrics.accuracy_score(Y_test,Y_pred)


# In[28]:


X=iris.iloc[:,2:4]
X


# In[29]:


Y=iris.iloc[:,4]


# In[30]:


Y.head()


# In[31]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15,random_state=1)


# In[32]:


model.fit(X_test,Y_test)


# In[33]:


Y_pred=model.predict(X_test)


# In[34]:


metrics.accuracy_score(Y_test,Y_pred)


# #OVO AND OVR

# In[70]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
ovr=OneVsRestClassifier(LogisticRegression()).fit(X_train,Y_train)
ovo=OneVsOneClassifier(LogisticRegression()).fit(X_train,Y_train)


# In[39]:


ovr.coef_


# In[40]:


ovr.intercept_


# In[42]:


metrics.accuracy_score(Y_train,ovr.predict(X_train))


# In[43]:


metrics.accuracy_score(Y_train,ovo.predict(X_train))


# In[45]:


metrics.accuracy_score(Y_test,ovr.predict(X_test))


# In[46]:


metrics.accuracy_score(Y_test,ovo.predict(X_test))


# #MOTIVATION FOR SVM

# In[57]:


pi=np.random.uniform(0,6.14,100)
le=np.random.uniform(0,2,100)
x1=list(le*np.cos(pi))
y1=list(le*np.sin(pi))
c1=[1]*100

pi=np.random.uniform(0,6.14,100)
le=np.random.uniform(4,6,100)
x2=list(le*np.cos(pi))
y2=list(le*np.sin(pi))
c2=[2]*100

pi=np.random.uniform(0,6.14,100)
le=np.random.uniform(8,10,100)
x3=list(le*np.cos(pi))
y3=list(le*np.sin(pi))
c3=[3]*100


# In[58]:


plt.scatter(x1,y1,color='red')
plt.scatter(x2,y2,color='blue')
plt.scatter(x3,y3,color='black')


# In[59]:


data=pd.DataFrame(columns=['f1','f2','class'])
data.f1=x1+x2+x3
data.f2=y1+y2+y3
data['class']=c1+c2+c3


# In[60]:


data.head()


# In[62]:


l=LogisticRegression()
l.fit(data.iloc[:,:2],data.iloc[:,2])


# In[63]:


metrics.accuracy_score(data.iloc[:,2],l.predict(data.iloc[:,:2]))


# In[76]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
ovr=OneVsRestClassifier(LogisticRegression()).fit(data.iloc[:,:2],data.iloc[:,2])
ovo=OneVsOneClassifier(LogisticRegression()).fit(data.iloc[:,:2],data.iloc[:,2])


# In[77]:


metrics.accuracy_score(data.iloc[:,2],ovo.predict(data.iloc[:,:2]))


# In[78]:


metrics.accuracy_score(data.iloc[:,2],ovr.predict(data.iloc[:,:2]))


# In[ ]:





# #DIABETES DATA

# In[81]:


df=pd.read_csv('diabetes.csv')
df.head()


# In[85]:


df.isnull().sum()


# In[89]:


x=df.drop('Outcome',axis=1)
y=df['Outcome']


# In[90]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15,random_state=1)


# In[91]:


l.fit(X_train,Y_train)


# In[92]:


w=l.predict(X_test)


# In[95]:


metrics.accuracy_score(Y_test,w)


# In[98]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
ovr=OneVsRestClassifier(LogisticRegression()).fit(X_train,Y_train)
ovo=OneVsOneClassifier(LogisticRegression()).fit(X_train,Y_train)


# In[99]:


metrics.accuracy_score(Y_test,ovr.predict(X_test))


# In[100]:


metrics.accuracy_score(Y_test,ovo.predict(X_test))


# In[ ]:




