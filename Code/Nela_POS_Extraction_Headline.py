import nltk
import csv
nltk.download('treebank')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import os

path="D:\Semester 2\Information Retreival\Assignment_1\Old_Raw"
os.chdir(path)

files=os.listdir()
files

import numpy as np
import pickle
import pandas as pd 
import seaborn as sns
import re
from collections import Counter
from tqdm.notebook import tqdm
from nltk.corpus import treebank
import matplotlib.pyplot as plt




pos_dict = {}

tag_set = set([tag for sent in treebank.tagged_sents() for _, tag in sent if any(char.isalpha() for char in tag)])

for index, tag in enumerate(tag_set):
    pos_dict[tag] = index


with open('pos_dict.pkl', 'wb') as f:
    pickle.dump(pos_dict, f)




pos_pattern_dict = {}
from itertools import product
pairs = list(product(range(len(pos_dict)), repeat=2))
for index, pair in enumerate(pairs):
    pos_triplet = (pair[0], pos_dict['CD'], pair[1])
    pos_pattern_dict[pos_triplet] = index






def extract_pos_patterns(sentence):
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    pos_patterns = []
    for i in range(len(pos_tags) - 2):
        if pos_tags[i+1][1] == 'CD':
            index = (i, i+1, i+2)
            # print(pos_tags[i][0], pos_tags[i+1][0], pos_tags[i+2][0])
            # print(pos_tags[i][1], pos_tags[i+1][1], pos_tags[i+2][1])
            if (pos_tags[i][1] in pos_dict) and (pos_tags[i+1][1] in pos_dict) and (pos_tags[i+2][1] in pos_dict):
                tags = (pos_dict[pos_tags[i][1]], pos_dict[pos_tags[i+1][1]], pos_dict[pos_tags[i+2][1]])
                # print(index, words, tags)
                pos_patterns.append((index, pos_pattern_dict[tags]))
    return pos_patterns



train_file = 'nela_preprocessed_train_posat.csv'
train_df = pd.read_csv(train_file)
train_df.dropna(inplace=True)

test_file = 'nela_preprocessed_test_posat.csv'
test_df = pd.read_csv(test_file)
test_df.dropna(inplace=True)




train_headline = train_df['Headline'].values
train_body = train_df['Body'].values
train_stance = train_df['Label'].values
test_headline = test_df['Headline'].values
test_body = test_df['Body'].values
test_stance = test_df['Label'].values



x_train_headline = []
x_train_cardinal_phrase = []
x_train_pos_pattern = []
x_train_body = []
y_train = []
for index, headline in enumerate(train_headline):
    pos_patterns = extract_pos_patterns(headline)
    if pos_patterns==[]:
        x_train_headline.append(headline)
        x_train_cardinal_phrase.append((0,0,0))  # Append(0,0,0) if no cardnial phrase
        x_train_pos_pattern.append(-1) # Append -1 if no pos tag found
        x_train_body.append(train_body[index])
        y_train.append(train_stance[index])
    else:
        for pos_pattern in pos_patterns:
            x_train_headline.append(headline)
            x_train_cardinal_phrase.append(pos_pattern[0])
            x_train_pos_pattern.append(pos_pattern[1])
            x_train_body.append(train_body[index])
            y_train.append(train_stance[index])




x_test_headline = []
x_test_cardinal_phrase = []
x_test_pos_pattern = []
x_test_body = []
y_test = []
for index, headline in enumerate(test_headline):
    pos_patterns = extract_pos_patterns(headline)
    if pos_patterns==[]:
        x_test_headline.append(headline)
        x_test_cardinal_phrase.append((0,0,0)) # Append(0,0,0) if no cardnial phrase
        x_test_pos_pattern.append(-1) # Append -1 if no pos tag found
        x_test_body.append(test_body[index])
        y_test.append(test_stance[index])
    for pos_pattern in pos_patterns:
        x_test_headline.append(headline)
        x_test_cardinal_phrase.append(pos_pattern[0])
        x_test_pos_pattern.append(pos_pattern[1])
        x_test_body.append(test_body[index])
        y_test.append(test_stance[index])





review_len = [i.count(" ") for i in x_train_headline]
pd.Series(review_len).hist()
plt.title("Number of words in Train Headline")
plt.show()
pd.Series(review_len).describe()



review_len = [i.count(" ") for i in x_train_body]
pd.Series(review_len).hist()
plt.title("Number of words in Train Body")
plt.show()
pd.Series(review_len).describe()



with open('train_pos_extracted.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Headline', 'Cardinal-phrase', 'POS-pattern', 'Body', 'Label'])
    writer.writerows(zip(x_train_headline, x_train_cardinal_phrase, x_train_pos_pattern, x_train_body, y_train))

with open('test_pos_extracted.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Headline', 'Cardinal-phrase', 'POS-pattern', 'Body', 'Label'])
    writer.writerows(zip(x_test_headline, x_test_cardinal_phrase, x_test_pos_pattern, x_test_body, y_test))

train_df=pd.read_csv("nela_pos_extracted_train.csv")
test_df=pd.read_csv("nela_pos_extracted_test.csv")

test_df.loc[test_df['POS-pattern']!=-1,:].to_csv("nela_test_derived_pos_extracted.csv")


train_df.loc[train_df['POS-pattern']!=-1,:].to_csv("nela_train_derived_pos_extracted.csv")



test_df.loc[test_df['POS-pattern']!=-1,["Headline","Body","Label"]].drop_duplicates().to_csv("nela_test_derived_preprocessed.csv")


train_df.loc[train_df['POS-pattern']!=-1,["Headline","Body","Label"]].drop_duplicates().to_csv("nela_train_derived_preprocessed.csv")






