#!/usr/bin/env python
# coding: utf-8

# In[1]:


# topic modeling
###
###
###


# In[2]:


import pandas as pd
import numpy as np

# load the data
reviews_datasets = pd.read_csv('TMSMM_class6_Amazon_Reviews_small.csv')
reviews_datasets = reviews_datasets.head(450)
reviews_datasets.dropna()


# In[3]:


# inspect some data
reviews_datasets['Text;;'][433]


# In[4]:


# create vocabulary of all the words in our data
# remove stopwords
# only include those words that appear in less than 80% of the document and appear in at least 2 documents
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix = count_vect.fit_transform(reviews_datasets['Text;;'].values.astype('U'))


# In[5]:


# 450 documents, vocabluary of 216 words
doc_term_matrix


# In[6]:


# use LDA to create topics along with the probability distribution for each word in our vocabulary for each topic
from sklearn.decomposition import LatentDirichletAllocation

# parameter n_components specifies the number of categories, or topics, that we want our text to be divided into
# seed to get similar results
LDA = LatentDirichletAllocation(n_components=5, random_state=42)
LDA.fit(doc_term_matrix)


# In[7]:


# randomly fetch words from our vocabulary
# it randomly fetches 10 words from our vocabulary
import random

for i in range(10):
    random_id = random.randint(0,len(count_vect.get_feature_names()))
    print(count_vect.get_feature_names()[random_id])


# In[8]:


# find 10 words with the highest probability for the first topic
first_topic = LDA.components_[0]


# In[9]:


top_topic_words = first_topic.argsort()[-10:]


# In[10]:


for i in top_topic_words:
    print(count_vect.get_feature_names()[i])


# In[11]:


# 10 words with highest probabilities for all the five topics
for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[12]:


# add a column to the original data frame that will store the topic for the text
topic_values = LDA.transform(doc_term_matrix)
topic_values.shape


# In[13]:


# add a new column for topic in the data frame and assigns the topic value to each row in the column
reviews_datasets['Topic'] = topic_values.argmax(axis=1)

