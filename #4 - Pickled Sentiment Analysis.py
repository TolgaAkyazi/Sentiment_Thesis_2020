import numpy as np
import pandas as pd
import pickle
import os


## load in pickled predictions from negative sentiment script ##
y_pred_train = np.load('C:/Users/tolga/Desktop/SCRIPTS AND SHI/Important Scripts/Final Scripts/pickles/predicted.pickle', allow_pickle = True)
## load in pickled data to make sentiment predictions ##
test_data_tfidf = np.load('C:/Users/tolga/Desktop/Final Project/Final Scripts/pickles/testset_tfidf_vector.pickle', allow_pickle = True)
train_data_tfidf_positive = np.load('C:/Users/tolga/Desktop/Final Project/Final Scripts/pickles/trainset_tfidf_vector.pickle', allow_pickle = True)
train_data_tfidf_negative = np.load('C:/Users/tolga/Desktop/Final Project/Final Scripts/pickles/trainset_tfidf_vector.pickle', allow_pickle = True)
test_data_cleaned_transcripts = np.load('C:/Users/tolga/Desktop/Final Project/Final Scripts/pickles/transcripts_cleaned_test.pickle', allow_pickle = True)
train_data_cleaned_transcripts = np.load('C:/Users/tolga/Desktop/Final Project/Final Scripts/pickles/transcripts_cleaned_train.pickle', allow_pickle = True)
#########      predictions on training data     ########
### Making a Dataframe of the predictions per video, for easy accessibility ###
### These Prections will function as the bias in further classification ###

import pandas as pd
df = pd.DataFrame({
    'predictions': y_pred_train
})

## returns a list of indices of positive classes ##
## used to multiply TF-IDF vectors at the given indices ##
test_list_index = (df.index[df['predictions'] == False].tolist())
#weighting according to sentiment
weighting_factor = 3
for i in test_list_index:
    train_data_tfidf_positive[i] = weighting_factor * train_data_tfidf_positive[i]

X_train_sentiment = train_data_tfidf_positive
y_train_sentiment = []
###
# merge categories
for document in train_data_cleaned_transcripts:
    class_made = 0
    if str(document[1]) == "3":
        class_made = 1
    else:
        class_made = 0
    y_train_sentiment.append(class_made)

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
#X_test_sentiment = test_data_tfidf
#y_test_sentiment = []

# merge categories
#for document in test_data_cleaned_transcripts:
#    class_made = 0
#    if str(document[1]) == "3":
#        class_made = 1
#    else:
#        class_made = 0
#    y_test_sentiment.append(class_made)
#####################################################

###############################
## initializing the classifier ##
from sklearn.linear_model import SGDClassifier
Log_Reg = SGDClassifier(loss='log', random_state=123)
Log_Reg.fit(X_train_sentiment,y_train_sentiment)
##############################

# support vector machine
from sklearn.svm import LinearSVC

SVM = LinearSVC(C=2.7, random_state=123, tol=1e-5, class_weight={0: '1', 1: '2'})
SVM.fit(X_train_sentiment, y_train_sentiment)

# decision tree
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(random_state=123, class_weight={0: '1', 1: '2'})
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

import pandas as pd
df = pd.DataFrame({
    'predictions': y_pred_train
})

## returns a list of indices of positive classes ##
## used to multiply TF-IDF vectors at the given indices ##
test_list_index = (df.index[df['predictions'] == False].tolist())
#weighting according to sentiment
weighting_factor = 3
for i in test_list_index:
    train_data_tfidf_positive[i] = weighting_factor * train_data_tfidf_positive[i]

X_train_sentiment = train_data_tfidf_positive
y_train_sentiment = []
###
# merge categories
for document in train_data_cleaned_transcripts:
    class_made = 0
    if str(document[1]) == "3":
        class_made = 1
    else:
        class_made = 0
    y_train_sentiment.append(class_made)

## split for training set

from sklearn.model_selection import train_test_split
X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(X_train_sentiment, y_train_sentiment, test_size=0.2, random_state=0)

###############################

#### testing phase
X_test_sentiment = test_data_tfidf
y_test_sentiment = []
####

for document in test_data_cleaned_transcripts:
    class_made = 0
    if str(document[1]) == "3":
        class_made = 1
    else:
        class_made = 0
    y_test_sentiment.append(class_made)
## load in cleaned transcripts from previous script


# merge categories


## initializing the classifier ##
from sklearn.linear_model import SGDClassifier
Log_Reg = SGDClassifier(loss='log', random_state=123)
Log_Reg.fit(X_train_sentiment,y_train_sentiment)
##############################

#support vector machine
from sklearn.svm import LinearSVC
SVM = LinearSVC(random_state=123)
SVM.fit(X_train_sentiment,y_train_sentiment)

#decision tree
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(random_state=123)
DT.fit(X_train_sentiment,y_train_sentiment)

#naive bayes
from sklearn.naive_bayes import BernoulliNB
NB = BernoulliNB()
NB.fit(X_train_sentiment,y_train_sentiment)

## Final reports ##
final_y_pred_log = Log_Reg.predict(X_test_sentiment)
final_y_pred_log2 = SVM.predict(X_test_sentiment)
final_y_pred_log3 = DT.predict(X_test_sentiment)
final_y_pred_log5 = NB.predict(X_test_sentiment)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('Logistic Regression + Positive Sentimental Bias: ')
print(confusion_matrix(y_test_sentiment,final_y_pred_log))
print(classification_report(y_test_sentiment,final_y_pred_log, digits=4))
print(accuracy_score(y_test_sentiment, final_y_pred_log))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('Support Vector Machine + Positive Sentimental Bias: ')
print(confusion_matrix(y_test_sentiment,final_y_pred_log2))
print(classification_report(y_test_sentiment,final_y_pred_log2, digits=4))
print(accuracy_score(y_test_sentiment, final_y_pred_log2))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('Decision Tree + Positive Sentimental Bias: ')
print(confusion_matrix(y_test_sentiment,final_y_pred_log3))
print(classification_report(y_test_sentiment,final_y_pred_log3, digits=4))
print(accuracy_score(y_test_sentiment, final_y_pred_log3))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('Naive Bayes + Positive Sentimental Bias: ')
print(confusion_matrix(y_test_sentiment,final_y_pred_log5))
print(classification_report(y_test_sentiment,final_y_pred_log5, digits=4))
print(accuracy_score(y_test_sentiment, final_y_pred_log5))