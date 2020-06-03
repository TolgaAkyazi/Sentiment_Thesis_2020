## load in all the necessary data ##
ourwatched_set_filepath = 'C:/Users/tolga/Desktop/SCRIPTS AND SHI/Resources'
pickle_filepath = 'C:/Users/tolga/Desktop/SCRIPTS AND SHI/Important Scripts/Final Scripts/pickles'
dataset_filepath = 'C:/Users/tolga/Desktop/SCRIPTS AND SHI/Resources'
transcript_filepath = 'C:/Users/tolga/Desktop/SCRIPTS AND SHI/Resources'

## shuffle datasets ##
## The training and test data both consist of the shape ([transcript, conspiracy rating]) ##
from stratifier import dataset_shuffler
test_data, training_data = dataset_shuffler(ourwatched_set_filepath, pickle_filepath, dataset_filepath, transcript_filepath)

# THE NLP
# importing spacy packages
import en_core_web_sm
nlp = en_core_web_sm.load(disable=["parser", "tagger", "ner"])
transcripts_cleaned = []
# defining the cleaner (lemmatization, tokenization, stopword removal, remove small tokens, decapitalization and alphanumerical only)
def my_cleaner3(text):
    return[token.lemma_.lower() for token in nlp(text) if not (token.is_stop or token.is_alpha==False or len(token.lemma_)<3)]

##############################
transcripts_cleaned_train = []
for i, transcript in enumerate(training_data):
    doc = nlp(str(training_data[i][0]))
    cleaned_tokens1 = my_cleaner3(doc.text)
    transcripts_cleaned_train.append([cleaned_tokens1,training_data[i][1]])

transcripts_cleaned_test = []
for i, transcript in enumerate(test_data):
    doc = nlp(str(test_data[i][0]))
    cleaned_tokens1 = my_cleaner3(doc.text)
    transcripts_cleaned_test.append([cleaned_tokens1,test_data[i][1]])

trainset_document_tokens = []
for i, document in enumerate(transcripts_cleaned_train):
    trainset_document_tokens.append(transcripts_cleaned_train[i][0])

testset_document_tokens = []
for i, document in enumerate(transcripts_cleaned_test):
    testset_document_tokens.append(transcripts_cleaned_test[i][0])

full_document_tokens = []

for i,doc in enumerate(trainset_document_tokens):
    for j in doc:
        full_document_tokens.append(j)

for i,doc in enumerate(testset_document_tokens):
    for j in doc:
        full_document_tokens.append(j)

########################################## EDA ##########################################
## Get counts per category
labels_train = {'1':0, '2':0, '3':0}
for i,doc in enumerate(transcripts_cleaned_train):
    if transcripts_cleaned_train[i][1] == 1:
        labels_train['1'] += 1
    if transcripts_cleaned_train[i][1] == 2:
        labels_train['2'] += 1
    if transcripts_cleaned_train[i][1] == 3:
        labels_train['3'] += 1

## Get counts per category
labels_test = {'1':0, '2':0, '3':0}
for i,doc in enumerate(transcripts_cleaned_test):
    if transcripts_cleaned_test[i][1] == 1:
        labels_test['1'] += 1
    if transcripts_cleaned_test[i][1] == 2:
        labels_test['2'] += 1
    if transcripts_cleaned_test[i][1] == 3:
        labels_test['3'] += 1

### PLOT FOR FREQUENCIES ###
import matplotlib.pyplot as plt
plt.bar(labels_test.keys(), labels_test.values(), color=['g', 'r', 'b'])
plt.xticks(range(len(labels_test.values())), labels_test.keys(), size='small')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.title('Histogram: Label distribution')
plt.show()
########################################## EDA ##########################################

import numpy as np
## TOTAL number of videos
sum_of_vids = sum(labels_train.values()) + sum(labels_test.values())
## TOTAL word count in whole dataset
total_word_count = len(full_document_tokens)
## Mean of word counts in whole dataset
mean_word_count = total_word_count/sum_of_vids
## Standard deviation in whole dataset
stdev_count = []
for i,doc in enumerate(trainset_document_tokens):
    stdev_count.append(len(trainset_document_tokens[i]))
for i,doc in enumerate(testset_document_tokens):
    stdev_count.append(len(testset_document_tokens[i]))
stdev = np.std(stdev_count)
print('Total word count: ', '\t', '\t' ,total_word_count, '\n', 'Mean word count: ','\t', '\t' ,round(mean_word_count,2),'\n', 'Standard deviation: ','\t' , round(stdev,2))






import pandas as pd
df = pd.DataFrame(stdev_count)
## adjust to your own directory ##################################
directory_for_csv = 'C:/Users/tolga/Desktop/SCRIPTS AND SHI/k.csv'
##################################################################
df.to_csv(directory_for_csv, index=False)

## unique word counts
from collections import Counter

keys = Counter(full_document_tokens).keys() # equals to list(set(words))
values = Counter(full_document_tokens).values() # counts the elements' frequency

print('Unique Baseline Features: ',len(values))

## Get word frequencies above 4000 ##
counts = dict()
for l,p in enumerate(trainset_document_tokens):
    for j in trainset_document_tokens[l]:
        counts[j] = counts.get(j, 0) + 1
for l,p in enumerate(testset_document_tokens):
    for j in testset_document_tokens[l]:
        counts[j] = counts.get(j, 0) + 1

value = []
for j,k in zip(counts.values(), counts.keys()):
    value.append([j,k])
for e,r in enumerate(value):
    if value[e][0] > 4000:
        print([value[e][0], value[e][1]])

### EDA for IMDB reviews dataset ###
# LOAD IMDB DATA
directory_pos = 'C:/Users/tolga/Desktop/SCRIPTS AND SHI/IMDB/train/pos'
directory_neg = 'C:/Users/tolga/Desktop/SCRIPTS AND SHI/IMDB/train/neg'
import os
pos_list = []
neg_list = []
from SentimentFunctions import SentimentExtractor
Sentiment_ = SentimentExtractor(directory_pos,directory_neg)
pos_list = Sentiment_[0]
neg_list = Sentiment_[1]

### NLP ### With the help of Raf van den Eijnden and Siebe Albers ###
#import necessary modules for NLP

## requires nlp + mycleaner3 to be ran in the previous part of the script ##
## does not work if ran seperately ##

# Cleaning the sentiment review files
sentiment_cleaned = []
for i, transcript in enumerate(pos_list):
    doc = nlp(pos_list[i][0])
    cleaned_tokens1 = my_cleaner3(doc.text)
    sentiment_cleaned.append(cleaned_tokens1)

for i, transcript in enumerate(neg_list):
    doc = nlp(neg_list[i][0])
    cleaned_tokens2 = my_cleaner3(doc.text)
    sentiment_cleaned.append(cleaned_tokens2)

### TF-IDF input, usable data for the classifiers ###
sentiment_tfidf_input = []
for document in sentiment_cleaned:
    sentiment_tfidf_input.append(" ".join(document))

from collections import Counter

words = ['a', 'b', 'c', 'a']

Counter(words).keys() # equals to list(set(words))
Counter(words).values() # counts the elements' frequency

full_sentiment = []
for i,doc in enumerate(sentiment_cleaned):
    for j in doc:
        full_sentiment.append(j)

keys_sentiment = len(Counter(full_sentiment).keys())
values_sentiment = Counter(full_sentiment).values() # counts the elements' frequency

print('Unique Sentiment Features:',keys_sentiment)

