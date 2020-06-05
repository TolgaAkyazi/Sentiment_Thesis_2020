import numpy as np
import pandas as pd
import pickle
## load in all the necessary data ## Set the files accordingly to your own directories ##
directory_pos = 'C:/Users/tolga/Desktop/Final Project/Final Scripts/Resource Files/Sentiment Resources/pos'
directory_neg = 'C:/Users/tolga/Desktop/Final Project/Final Scripts/Resource Files/Sentiment Resources/neg'

import os
pos_list = []
neg_list = []

from SentimentFunctions import SentimentExtractor
Sentiment_ = SentimentExtractor(directory_pos,directory_neg)
pos_list = Sentiment_[0]
neg_list = Sentiment_[1]

### NLP ### With the help of Raf van den Eijnden and Siebe Albers ###
#import necessary modules for NLP
import en_core_web_sm
nlp = en_core_web_sm.load(disable=["parser", "tagger", "ner"])
# defining the NLP cleaner (lemmatization, tokenization, stopword removal, remove small tokens, decapitalization and alphanumerical only)
def my_cleaner3(text):
    return[token.lemma_.lower() for token in nlp(text) if not (token.is_stop or token.is_alpha==False or len(token.lemma_)<3)]

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

####################################### TF_IDF vectorizer ######################################
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(max_features=5000, encoding='utf-8')
################################################################################################

## TF-IDF vectors of IMDB reviews ##
sentiment_tfidf_vector = tf.fit_transform(sentiment_tfidf_input).toarray()

### MODEL TRAINING
X_train = sentiment_tfidf_vector
y_train = []

for i, document in enumerate(pos_list):
    y_train.append(pos_list[i][1])

for i, document in enumerate(neg_list):
    y_train.append(neg_list[i][1])

##                                                                        ##
## Logistic Regression. Fit the model on Sentiment Data of IMDB reviews   ##
##                                                                        ##

from sklearn.linear_model import SGDClassifier
Log_Reg = SGDClassifier(loss='log', random_state=123)
Log_Reg.fit(X_train,y_train)

####### Sentiment Classifier has been trained,
####### now its time for some unsupervised Sentiment Classification on the Youtube Transcripts

### With our last script we made pickled test and training files ###

## load in pickled data to make sentiment predictions ##
test_data_tfidf = np.load('C:/Users/tolga/Desktop/Final Project/Final Scripts/pickles/testset_tfidf_vector.pickle', allow_pickle = True)
train_data_tfidf_negative = np.load('C:/Users/tolga/Desktop/Final Project/Final Scripts/pickles/trainset_tfidf_vector.pickle', allow_pickle = True)
test_data_cleaned_transcripts = np.load('C:/Users/tolga/Desktop/Final Project/Final Scripts/pickles/transcripts_cleaned_test.pickle', allow_pickle = True)
train_data_cleaned_transcripts = np.load('C:/Users/tolga/Desktop/Final Project/Final Scripts/pickles/transcripts_cleaned_train.pickle', allow_pickle = True)
#########      predictions on training data     ########
y_pred_train = Log_Reg.predict(train_data_tfidf_negative)

## filepath of: which folder the pickled data should be stored
pickle_filepath = 'C:\\Users\\tolga\\Desktop\\Final Project\\Final Scripts\\pickles'
# pickle the predictions on training data #
os.chdir(pickle_filepath)
with open('predicted.pickle', 'wb') as f:
    pickle.dump(y_pred_train, f)

### Making a Dataframe of the predictions per video, for easy accessibility ###
### These Prections will function as the bias in further classification ###
df = pd.DataFrame({
    'predictions': y_pred_train
    })

## returns a list of indices of negative classes ##
## used to multiply TF-IDF vectors at the given indices ##
test_list_index = (df.index[df['predictions'] == True].tolist())
#weighting according to sentiment
weighting_factor = 3
for i in test_list_index:
    train_data_tfidf_negative[i] = weighting_factor * train_data_tfidf_negative[i]

X_train_sentiment = train_data_tfidf_negative
y_train_sentiment = []

# merge categories
for document in train_data_cleaned_transcripts:
    class_made = 0
    if str(document[1]) == "3":
        class_made = 1
    else:
        class_made = 0
    y_train_sentiment.append(class_made)

### training split ###
from sklearn.model_selection import train_test_split
X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(X_train_sentiment, y_train_sentiment, test_size=0.2, random_state=0)
######################

################ only for testing phase ##############
X_test_sentiment = test_data_tfidf
y_test_sentiment = []

# merge categories
for document in test_data_cleaned_transcripts:
    class_made = 0
    if str(document[1]) == "3":
        class_made = 1
    else:
        class_made = 0
    y_test_sentiment.append(class_made)
#####################################################

###############################
## initializing the classifier ##
from sklearn.linear_model import SGDClassifier
Log_Reg = SGDClassifier(loss='log', random_state=123)
Log_Reg.fit(X_train_sentiment,y_train_sentiment)
##############################

# support vector machine
from sklearn.svm import LinearSVC

SVM = LinearSVC(random_state=123)
SVM.fit(X_train_sentiment, y_train_sentiment)

# decision tree
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(random_state=123)
DT.fit(X_train_sentiment, y_train_sentiment)

# naive bayes
from sklearn.naive_bayes import BernoulliNB

NB = BernoulliNB()
NB.fit(X_train_sentiment, y_train_sentiment)

## Final reports ##
final_y_pred_log = Log_Reg.predict(X_test_sentiment)
final_y_pred_log2 = SVM.predict(X_test_sentiment)
final_y_pred_log3 = DT.predict(X_test_sentiment)
final_y_pred_log5 = NB.predict(X_test_sentiment)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('Logistic Regression + Negative Sentimental Bias: ')
print(confusion_matrix(y_test_sentiment, final_y_pred_log))
print(classification_report(y_test_sentiment, final_y_pred_log, digits=4))
print(accuracy_score(y_test_sentiment, final_y_pred_log))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('Support Vector Machine + Negative Sentimental Bias: ')
print(confusion_matrix(y_test_sentiment, final_y_pred_log2))
print(classification_report(y_test_sentiment, final_y_pred_log2, digits=4))
print(accuracy_score(y_test_sentiment, final_y_pred_log2))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('Decision Tree + Negative Sentimental Bias: ')
print(confusion_matrix(y_test_sentiment, final_y_pred_log3))
print(classification_report(y_test_sentiment, final_y_pred_log3, digits=4))
print(accuracy_score(y_test_sentiment, final_y_pred_log3))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('Naive Bayes + Negative Sentimental Bias: ')
print(confusion_matrix(y_test_sentiment, final_y_pred_log5))
print(classification_report(y_test_sentiment, final_y_pred_log5, digits=4))
print(accuracy_score(y_test_sentiment, final_y_pred_log5))