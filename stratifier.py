import os

def dataset_shuffler(ourwatched_set_filepath, pickle_filepath, dataset_filepath,transcript_filepath):
    """
    This function collects our transcript data from Marks set and ours, combines it into one
    then stratified-splits it into test_data and training_data.
    :param ourwatched_set_filepath: input the folder where our_watchedset.csv is located
    :param pickle_filepath: input the folder path where you want to save the training and test set
    :param dataset_filepath: input the folder path where the YouTube.csv is located
    :param transcript_filepath: input the folder path where the JSONs of transcripts are located
    :return: returns two variables, one with test, one with training data, can be called like this:

    test_data, training_data = dataset_shuffler('D:/School/CSAI/Thesis/Data Exploration Project','D:/School/CSAI/Thesis/Data Exploration Project','D:/School/CSAI/Thesis/Dataset','D:/School/CSAI/Thesis/Dataset/Transcripts')

    also returns two picklefiles in the specified folder containing the same sets.
    """
    import os
    import csv
    import numpy as np
    import pickle
    from my_FunctionsModule_ForThesis import TranscriptExtractor
    Transcripts = TranscriptExtractor(dataset_filepath,transcript_filepath)
    blabla = Transcripts

    blabla.sort()

    bla = []
    for item in blabla:
        bla.append([item[2],item[3]])
    new_k = []
    for elem in bla:
        if elem not in new_k:
            new_k.append(elem)
    bla = new_k

    ourwatchedset = []
    os.chdir(ourwatched_set_filepath)
    with open(ourwatched_set_filepath+'/Transcripts2nddataset.csv', 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for i,row in enumerate(csv_reader):
            if i == 0:
                continue
            else:
                ourwatchedset.append(row)

    os.chdir(ourwatched_set_filepath)

    combined = []
    for file in new_k:
        combined.append([file[0],file[1]])

    for transcript in ourwatchedset:
        combined.append([transcript[1],transcript[2]])
    X = []
    y = []
    for item in combined:
        if item[1] != 'x':
            X.append(item[0])
        if item[1] != 'x':
            y.append(int(item[1]))
    X = np.array(X)
    y = np.array(y)

    from sklearn.model_selection import StratifiedShuffleSplit
    indiced = StratifiedShuffleSplit(n_splits=1,train_size=0.80,random_state=0)

    for train_index,test_index in indiced.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    training_data = []

    test_data = []

    for i,item in enumerate(X_train):
        training_data.append([X_train[i],y_train[i]])
    for i,item in enumerate(X_test):
        test_data.append([X_test[i],y_test[i]])

    import pickle
    os.chdir(pickle_filepath)
    with open('training_data.pickle', 'wb') as f:
        pickle.dump(training_data, f)
    with open('test_data.pickle', 'wb') as f:
        pickle.dump(test_data, f)
    return test_data, training_data