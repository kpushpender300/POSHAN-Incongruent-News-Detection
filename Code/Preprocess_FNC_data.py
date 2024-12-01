#!/usr/bin/env python
# coding: utf-8

# In[24]:


import nltk
nltk.download('treebank')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# In[25]:


from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np 
import pandas as pd 
import seaborn as sns
import re
from nltk.corpus import stopwords
from collections import Counter
from tqdm.notebook import tqdm
import csv
import matplotlib.pyplot as plt
import os
import inflect


# In[26]:


def number_to_text(number):
    p = inflect.engine()
    text = p.number_to_words(number)
    return text


# In[27]:


lemmatizer = WordNetLemmatizer()
def preprocess(text):
    List=text.split()
    for i,words in enumerate(List):
        if words.isdigit():
            try:
                List[i] = number_to_text(int(words))
            except:
                List[i]=words
    text=" ".join(List)
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation marks and special characters
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Tokenize the text into individual words
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in (stop_words)]
    # Lemmatization
  
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in ["eos","eop"]]
    # Join the tokens back into a single string
    processed_text = ' '.join(tokens)
    # # Replace double spaces with single spaces
    processed_text = re.sub(r'\s+', ' ', processed_text)

    return processed_text

train_file = 'train_data_set.csv'
train_df = pd.read_csv(train_file)


# In[31]:


test_file = 'test_data_set.csv'
test_df = pd.read_csv(test_file)


# In[33]:


train_headline = train_df['Headline'].values
train_body = train_df['articleBody'].values
y_train = train_df['Stance'].values


# In[32]:


test_headline = test_df['Headline'].values
test_body = test_df['articleBody'].values
y_test = test_df['Stance'].values



import nltk
nltk.download('omw-1.4')



x_train_headline = []
for sen in train_headline:
  x_train_headline.append(preprocess(sen))

x_train_body = []
for sen in train_body:
  x_train_body.append(preprocess(sen))




x_test_headline = []
for sen in test_headline:
  x_test_headline.append(preprocess(sen))

x_test_body = []
for sen in test_body:
  x_test_body.append(preprocess(sen))





with open('preprocessed_fnc_train.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Headline', 'Body', 'Label'])
    writer.writerows(zip(x_train_headline, x_train_body, y_train))

with open('preprocessed_fnc_test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Headline', 'Body', 'Label'])
    writer.writerows(zip(x_test_headline, x_test_body, y_test))






