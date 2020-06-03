def Binarizer_new(Y):
    from collections import Counter
    COUNT_before = (Counter(Y))
    for I,J in enumerate(Y):
        if J == 1:
            Y[I] = 0
        elif J==2:
            Y[I] = 1
        elif J==3:
            Y[I] = 1
        else:
            'WE GOT A PROBLEM'
    COUNT_after = (Counter(Y))
    print('before',COUNT_before, '\nafter:',COUNT_after)

Binarizer_new(y_test)