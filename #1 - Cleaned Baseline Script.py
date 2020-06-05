import numpy as np
import pandas as pd
import pickle
import os
import json
import sys

###pycharm being a dick

###sys.path.insert(0,'C:/Users/tolga/Desktop/Final Project/Final Scripts')

#######################


### TO RUN FILES, THE FILEPATHS SHOULD BE SET TO YOUR PERSONAL DIRECTORIES ###
### IM NO PYTHON EXPERT, SORRY ##


## load in all the necessary data ## Set the files accordingly to your own directories ##
## filepath of: transcripts of ~50 videos we watched ourselves
ourwatched_set_filepath = 'C:/Users/tolga/Desktop/Final Project/Final Scripts/Resource Files/Baseline Resources'
## filepath of: which folder the pickled data should be stored
pickle_filepath = 'C:/Users/tolga/Desktop/Final Project/Final Scripts/pickles'
## filepath of: labels of transcripts from ~480 videos (Alfano, et al.)
dataset_filepath = 'C:/Users/tolga/Desktop/Final Project/Final Scripts/Resource Files/Baseline Resources'
## filepath of: transcripts from ~480 videos from (Alfano, et al.)
transcript_filepath = 'C:/Users/tolga/Desktop/Final Project/Final Scripts/Resource Files/Baseline Resources'

## shuffle datasets ##
## The training and test data both consist of the shape ([transcript, conspiracy rating]) ##
from stratifier import dataset_shuffler
test_data, training_data = dataset_shuffler(ourwatched_set_filepath, pickle_filepath, dataset_filepath, transcript_filepath)

## We now have a training and a test set consisting of data collected by Mark Alfano + our own collected data ##
## Now for the necessary NLP ## Both for the training and test data

## Importing SpaCy NLP modules ## With the help of Raf van den Eijnden and Siebe Albers ###

############################################# NLP #############################################
import en_core_web_sm
nlp = en_core_web_sm.load(disable=["parser", "tagger", "ner"])
# defining the NLP cleaner (lemmatization, tokenization, stopword removal, remove small tokens, decapitalization and alphanumerical only)
def my_cleaner3(text):
    return[token.lemma_.lower() for token in nlp(text) if not (token.is_stop or token.is_alpha==False or len(token.lemma_)<3)]
###############################################################################################

## Cleaning the transcripts
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
###############################################################################################

## Tokeninzing both the training and test set ##
## This is done so the data can be transformed into vectors ##
## For this algorithm we are going to use TF-IDF vectors to feed into the classifiers ##

###############################################################################################
trainset_document_tokens = []
for i, document in enumerate(transcripts_cleaned_train):
    trainset_document_tokens.append(transcripts_cleaned_train[i][0])

trainset_tfidf_input = []
for document in trainset_document_tokens:
    trainset_tfidf_input.append(" ".join(document))

testset_document_tokens = []
for i, document in enumerate(transcripts_cleaned_test):
    testset_document_tokens.append(transcripts_cleaned_test[i][0])

testset_tfidf_input = []
for document in testset_document_tokens:
    testset_tfidf_input.append(" ".join(document))
###############################################################################################

## By joining the tokens, we create a single list filled with NLP'ed tokens per video-transcript ##
## Both for the test and training set ##

## To make the TF-IDF vector useable, we need to initialize a TF-IDF vectorizer ##
## And after that we need to fit the Vectorizer to the training data we made in the previous section ##

####################################### TF_IDF vectorizer #####################################
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(max_features=5000, encoding='utf-8')
trainset_tfidf_vector = tf.fit_transform(trainset_tfidf_input).toarray()
testset_tfidf_vector = tf.transform(testset_tfidf_input).toarray()
###############################################################################################

################# FOR EASE OF USE ##################
# pickle the train and testset TF-IDF vectors #
os.chdir(pickle_filepath)
with open('trainset_tfidf_vector.pickle', 'wb') as f:
    pickle.dump(trainset_tfidf_vector, f)
with open('testset_tfidf_vector.pickle', 'wb') as f:
    pickle.dump(testset_tfidf_vector, f)

# pickle the train and testset cleaned transcripts #
with open('transcripts_cleaned_train.pickle', 'wb') as f:
    pickle.dump(transcripts_cleaned_train, f)
with open('transcripts_cleaned_test.pickle', 'wb') as f:
    pickle.dump(transcripts_cleaned_test, f)
###################################################

## We now have a vector which can be fed into a classifier to be trained ##

## First we make lists of X_train and y_train, so they can be further split into useful training data ##
X_train = trainset_tfidf_vector
y_train = []

# Making classes of the labeled transcripts #
## classes 1 & 2 --> '0'       ##
## class 3 --> '1'             ##

for document in transcripts_cleaned_train:
    class_made = 0
    if str(document[1]) == "3":
        class_made = 1
    else:
        class_made = 0
    y_train.append(class_made)

########## used only in testing phase ##########
X_test = testset_tfidf_vector
y_test = []

for document in transcripts_cleaned_test:
    class_made = 0
    if str(document[1]) == "3":
        class_made = 1
    else:
        class_made = 0
    y_test.append(class_made)
############################################


## To train the classifier, we split the training data into training and validaton sets ##
### Training Split of Unbiased Biased Dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

## We finally reached the stage of training a classifier on the data we made ##

## initializing the classifier ##
from sklearn.linear_model import SGDClassifier
Log_Reg = SGDClassifier(loss='log', random_state=123)

## fitting classifier on our training data ##
Log_reg_fitted = Log_Reg.fit(X_train,y_train)


## pickle the fitted data + cross validation score
os.chdir(pickle_filepath)
with open('Log_reg_fitted.pickle', 'wb') as f:
    pickle.dump(Log_reg_fitted, f)
with open('Kfold_acc.pickle', 'wb') as f:
    pickle.dump(alternative_Kfold_mean, f)

#support vector machine
from sklearn.svm import LinearSVC
SVM = LinearSVC(random_state=123)
SVM.fit(X_train,y_train)

#decision tree
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(random_state=123)
DT.fit(X_train,y_train)

#naive bayes
from sklearn.naive_bayes import BernoulliNB
NB = BernoulliNB()
NB.fit(X_train,y_train)

y_pred_log = Log_Reg.predict(X_test)
y_pred_svm = SVM.predict(X_test)
y_pred_DT = DT.predict(X_test)
y_pred_NB = NB.predict(X_test)

###    validation score    ##
## 10-fold cross validation ##
from sklearn.model_selection import cross_val_score
cross_val = (cross_val_score(Log_reg_fitted, X_train, y_train, cv=10))
alternative_Kfold_mean = np.mean(cross_val)
print('Average validation score Log Reg: ',alternative_Kfold_mean,'\n', 'Validation score per fold: ','\n',cross_val)

### rest of the classifiers' K-fold validation scores ###
from sklearn.model_selection import cross_val_score
cross_val_SVM = (cross_val_score(SVM, X_train, y_train, cv=10))
alternative_Kfold_mean_SVM = np.mean(cross_val_SVM)
print('Average validation score SVM: ',alternative_Kfold_mean_SVM,'\n', 'Validation score per fold: ','\n',cross_val_SVM)
print()
from sklearn.model_selection import cross_val_score
cross_val_DT = (cross_val_score(DT, X_train, y_train, cv=10))
alternative_Kfold_mean_DT = np.mean(cross_val_DT)
print('Average validation score DT: ',alternative_Kfold_mean_DT,'\n', 'Validation score per fold: ','\n',cross_val_DT)
print()
from sklearn.model_selection import cross_val_score
cross_val_NB = (cross_val_score(NB, X_train, y_train, cv=10))
alternative_Kfold_mean_NB = np.mean(cross_val_NB)
print('Average validation score NB: ',alternative_Kfold_mean_NB,'\n', 'Validation score per fold: ','\n',cross_val_NB)
print()

## classification reports
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('Logistic Regression Baseline:')
print(confusion_matrix(y_test,y_pred_log))
print(classification_report(y_test,y_pred_log, digits=4))
print(accuracy_score(y_test, y_pred_log))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('Support Vector Machine Baseline:')
print(confusion_matrix(y_test,y_pred_svm))
print(classification_report(y_test,y_pred_svm, digits=4))
print(accuracy_score(y_test, y_pred_svm))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('Decision Tree Baseline:')
print(confusion_matrix(y_test,y_pred_DT))
print(classification_report(y_test,y_pred_DT, digits=4))
print(accuracy_score(y_test, y_pred_DT))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('Naive Bayes Baseline:')
print(confusion_matrix(y_test,y_pred_NB))
print(classification_report(y_test,y_pred_NB, digits=4))
print(accuracy_score(y_test, y_pred_NB))