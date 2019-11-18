#!/usr/bin/env python
# coding: utf-8

# # SKLearn - Naive Bayes

# In[3]:


from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


# In[4]:


dataset=datasets.load_iris()


# In[5]:


model=GaussianNB()
model.fit(dataset.data,dataset.target)


# In[6]:


print(model)


# In[7]:


expected=dataset.target
predicted=model.predict(dataset.data)


# In[8]:


print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected,predicted))


# In[ ]:





# In[ ]:




