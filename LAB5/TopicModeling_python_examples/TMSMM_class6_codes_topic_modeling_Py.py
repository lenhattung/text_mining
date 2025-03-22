#!/usr/bin/env python
# coding: utf-8

# In[1]:


# topic modeling
###
###
###


# In[2]:


# packages to store and manipulate data
import pandas as pd
import numpy as np

# plotting packages
import matplotlib.pyplot as plt

# model building package
import sklearn

# package to clean text
import re


# In[3]:


# upload the dataset
df = pd.read_csv('TMSMM_class6_climate_tweets.csv')


# In[4]:


# inspect the dataset
print(df)


# In[5]:


# look for retweets - tweets that started with the letters ‘RT’
# lambda functions and string comparisons to find the retweets
# two string variables for comparison
string1 = 'climate'
string2 = 'climb'


# In[6]:


# normal function example
def my_normal_function(x):
    return x**2 + 10
# lambda function example
my_lambda_function = lambda x: x**2 + 10


# In[7]:


# make a new column to highlight retweets
df['is_retweet'] = df['tweet'].apply(lambda x: x[:2]=='RT')
df['is_retweet'].sum()  # number of retweets


# In[8]:


# number of unique retweets
df.loc[df['is_retweet']].tweet.unique().size


# In[9]:


# 10 most repeated tweets
df.groupby(['tweet']).size().reset_index(name='counts')  .sort_values('counts', ascending=False).head(10)


# In[10]:


# number of times each tweet appears
counts = df.groupby(['tweet']).size()           .reset_index(name='counts')           .counts

# define bins for histogram
my_bins = np.arange(0,counts.max()+2, 1)-0.5

# plot histogram of tweet counts
plt.figure()
plt.hist(counts, bins = my_bins)
plt.xlabels = np.arange(1,counts.max()+1, 1)
plt.xlabel('copies of each tweet')
plt.ylabel('frequency')
plt.yscale('log', nonposy='clip')
plt.show()


# In[11]:


# who is being retweeted 
# who is being tweeted at/mentioned 
# what hashtags are being used

def find_retweeted(tweet):
    '''This function will extract the twitter handles of retweed people'''
    return re.findall('(?<=RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

def find_mentioned(tweet):
    '''This function will extract the twitter handles of people mentioned in the tweet'''
    return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)  

def find_hashtags(tweet):
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet) 


# In[12]:


# make new columns for retweeted usernames, mentioned usernames and hashtags
df['retweeted'] = df.tweet.apply(find_retweeted)
df['mentioned'] = df.tweet.apply(find_mentioned)
df['hashtags'] = df.tweet.apply(find_hashtags)


# In[13]:


print(df)


# In[14]:


# take the rows from the hashtag columns where there are actually hashtags
hashtags_list_df = df.loc[
                       df.hashtags.apply(
                           lambda hashtags_list: hashtags_list !=[]
                       ),['hashtags']]


# In[15]:


# create dataframe where each use of hashtag gets its own row
flattened_hashtags_df = pd.DataFrame(
    [hashtag for hashtags_list in hashtags_list_df.hashtags
    for hashtag in hashtags_list],
    columns=['hashtag'])


# In[16]:


flattened_hashtags_df.head(10)


# In[17]:


# number of unique hashtags
flattened_hashtags_df['hashtag'].unique().size


# In[18]:


# count of appearances of each hashtag
popular_hashtags = flattened_hashtags_df.groupby('hashtag').size()                                        .reset_index(name='counts')                                        .sort_values('counts', ascending=False)                                        .reset_index(drop=True)


# In[19]:


# number of times each hashtag appears
counts = flattened_hashtags_df.groupby(['hashtag']).size()                              .reset_index(name='counts')                              .counts

# define bins for histogram                              
my_bins = np.arange(0,counts.max()+2, 5)-0.5

# plot histogram of tweet counts
plt.figure()
plt.hist(counts, bins = my_bins)
plt.xlabels = np.arange(1,counts.max()+1, 1)
plt.xlabel('hashtag number of appearances')
plt.ylabel('frequency')
plt.yscale('log', nonposy='clip')
plt.show()


# In[20]:


# take hashtags which appear at least this amount of times
min_appearance = 10
# find popular hashtags - make into python set for efficiency
popular_hashtags_set = set(popular_hashtags[
                           popular_hashtags.counts>=min_appearance
                           ]['hashtag'])


# In[21]:


# make a new column with only the popular hashtags
hashtags_list_df['popular_hashtags'] = hashtags_list_df.hashtags.apply(
            lambda hashtag_list: [hashtag for hashtag in hashtag_list
                                  if hashtag in popular_hashtags_set])
# drop rows without popular hashtag
popular_hashtags_list_df = hashtags_list_df.loc[
            hashtags_list_df.popular_hashtags.apply(lambda hashtag_list: hashtag_list !=[])]


# In[22]:


# make new dataframe
hashtag_vector_df = popular_hashtags_list_df.loc[:, ['popular_hashtags']]

for hashtag in popular_hashtags_set:
    # make columns to encode presence of hashtags
    hashtag_vector_df['{}'.format(hashtag)] = hashtag_vector_df.popular_hashtags.apply(
        lambda hashtag_list: int(hashtag in hashtag_list))


# In[23]:


print(hashtag_vector_df)


# In[25]:


hashtag_vector_df.head(10)


# In[26]:


hashtag_matrix = hashtag_vector_df.drop('popular_hashtags', axis=1)


# In[27]:


# calculate the correlation matrix
correlations = hashtag_matrix.corr()


# In[28]:


print(correlations)


# In[29]:


# move to topic modeling


# In[30]:


import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


# In[31]:


def remove_links(tweet):
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links
    tweet = tweet.strip('[link]') # remove [links]
    return tweet

def remove_users(tweet):
    '''Takes a string and removes retweet and @user information'''
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove retweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove tweeted at
    return tweet


# In[32]:


my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

# cleaning master function
def clean_tweet(tweet, bigrams=False):
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = tweet.lower() # lower case
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) # strip punctuation
    tweet = re.sub('\s+', ' ', tweet) #remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet) # remove numbers
    tweet_token_list = [word for word in tweet.split(' ')
                            if word not in my_stopwords] # remove stopwords

    tweet_token_list = [word_rooter(word) if '#' not in word else word
                        for word in tweet_token_list] # apply word rooter
    if bigrams:
        tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                            for i in range(len(tweet_token_list)-1)]
    tweet = ' '.join(tweet_token_list)
    return tweet


# In[33]:


# clean the data
df['clean_tweet'] = df.tweet.apply(clean_tweet)


# In[35]:


df.head(5)


# In[36]:


from sklearn.feature_extraction.text import CountVectorizer

# the vectorizer object will be used to transform text to vector form
vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')

# apply transformation
tf = vectorizer.fit_transform(df['clean_tweet']).toarray()

# tf_feature_names tells us what word each column in the matric represents
tf_feature_names = vectorizer.get_feature_names()


# In[37]:


from sklearn.decomposition import LatentDirichletAllocation

# let us assume 10 topics
number_of_topics = 10

# define the random state so that this model is reproducible
model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)


# In[38]:


# apply the model
model.fit(tf)


# In[39]:


# take a look at the output
def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


# In[40]:


no_top_words = 10
display_topics(model, tf_feature_names, no_top_words)

