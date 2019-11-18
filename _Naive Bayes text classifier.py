#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes text classifier

# In[8]:


#Loading necessary libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_20newsgroups
data= fetch_20newsgroups()
data.target_names


# In[11]:


# Defining categories
categories=['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

#train the data on these categories
train=fetch_20newsgroups(subset='train', categories=categories)

#Testing the data for these categories
test=fetch_20newsgroups(subset='test', categories=categories)

#Printing training data
print(train.data[5])


# In[12]:


#import necessary packages

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# creating a model based on  multinomial naive bayes
model=make_pipeline(TfidfVectorizer(),MultinomialNB())

#Training the model with the train data
model.fit(train.data, train.target)

#creating labels for the test data
labels=model.predict(test.data)


# In[16]:


#Creating confusion matrix

from sklearn.metrics import confusion_matrix
mat= confusion_matrix(test.target,labels)
sns.heatmap(mat.T,square=True,annot=True,cbar=False,xticklabels=train.target_names,
           yticklabels=train.target_names)

#Plotting heatmap of confusin matrix
plt.xlabel('true label')
plt.ylabel('predcted label');
    


# In[19]:


#predicting category on new data based in trainned model

def predict_category(s, train=train, model=model):
    pred=model.predict([s])
    return train.target_names[pred[0]]


# In[25]:


predict_category('Jesus christ ')


# In[26]:


predict_category('Sending load to international space station ISS')


# In[27]:


predict_category('Auzuki hayausa is a very fast motorcycle')


# In[28]:


predict_category('Audy is bettter than BMW')


# In[29]:


predict_category('president of India')

