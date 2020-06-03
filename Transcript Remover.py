#========================================================================#
' Removing < 100 length trans:        '
#========================================================================#
import numpy as np
import pandas as pd
import pickle
import os

## load in all the necessary data ##
ourwatched_set_filepath = 'C:/Users/tolga/Desktop/SCRIPTS AND SHI/Resources'
pickle_filepath = 'C:/Users/tolga/Desktop/SCRIPTS AND SHI/Important Scripts/Final Scripts/pickles'
dataset_filepath = 'C:/Users/tolga/Desktop/SCRIPTS AND SHI/Resources'
transcript_filepath = 'C:/Users/tolga/Desktop/SCRIPTS AND SHI/Resources'

## shuffle datasets ##
## The training and test data both consist of the shape ([transcript, conspiracy rating]) ##
from stratifier import dataset_shuffler
test_data, training_data = dataset_shuffler(ourwatched_set_filepath, pickle_filepath, dataset_filepath, transcript_filepath)

x_train = []
y_train = []

x_test = []
y_test = []

for i,doc in enumerate(training_data):
    x_train.append(training_data[i][0])
    y_train.append(training_data[i][1])

for i,doc in enumerate(test_data):
    x_test.append(test_data[i][0])
    y_test.append(test_data[i][1])

# for the train set
training_list = []
counter = 0
for I,J,K in zip(x_train,x_train,y_train):
    training_list.append(((len(I.split()),K,J)))

#deleting the <100
for J,I in enumerate(training_list):
    if I[0] <100:
        training_list.pop(J)
        y_train.pop(J) # extra check
        counter +=1
        print('\nindex that is removed',J)
        print('\n\nentire example:',I)
print('\n\n this many deleted',counter)

#checking if everything went well:
labelListCheck = [I[1] for I in training_list]
print(labelListCheck == y_train)

# for the test set
test_list= []
counter = 0
for I,J,K in zip(x_test,x_test,y_test):
    test_list.append(((len(I.split()),K,J)))

for J,I in enumerate(test_list):
    if I[0] <100:
        test_list.pop(J)
        y_test.pop(J)
        counter +=1
        print('\nindex that is removed',J)
        print('\n\nentire example:',I)
print('\n\n this many deleted',counter)

#checking if everyihing went well:
labelListCheck = [I[1] for I in test_list]
print(labelListCheck == y_test)