#!/usr/bin/env python
# coding: utf-8

# In[1]:


# topic modeling
###
###
###


# In[2]:


# Non-Negative Matrix Factorization (NMF)


# In[3]:


import pandas as pd
import numpy as np

# load the data
reviews_datasets = pd.read_csv('TMSMM_class6_Amazon_Reviews_small.csv')
reviews_datasets = reviews_datasets.head(450)
reviews_datasets.dropna()


# In[4]:


# use TFIDF vectorizer since NMF works with TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix = tfidf_vect.fit_transform(reviews_datasets['Text;;'].values.astype('U'))


# In[5]:


# we can create a probability matrix that contains probabilities of all the words in the vocabulary for all the topics
from sklearn.decomposition import NMF

nmf = NMF(n_components=5, random_state=42)
nmf.fit(doc_term_matrix )


# In[7]:


# randomly get 5 words from our vocabulary
import random

for i in range(5):
    random_id = random.randint(0,len(tfidf_vect.get_feature_names()))
    print(tfidf_vect.get_feature_names()[random_id])


# In[8]:


# retrieve the probability vector of words for the first topic and will retrieve the indexes of the five words with the highest probabilities
first_topic = nmf.components_[0]
top_topic_words = first_topic.argsort()[-5:]


# In[9]:


# tfidf_vect object to retrieve the actual words
for i in top_topic_words:
    print(tfidf_vect.get_feature_names()[i])


# In[10]:


# print the five words with highest probabilities for each of the topics
for i,topic in enumerate(nmf.components_):
    print(f'Top five words for topic #{i}:')
    print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-5:]])
    print('\n')


# In[11]:


# adds the topics to the data set and displays the first five rows
topic_values = nmf.transform(doc_term_matrix)
reviews_datasets['Topic'] = topic_values.argmax(axis=1)

