import os

pos_list = []
neg_list = []

def SentimentExtractor(directory_pos, directory_neg):
    for root, dirs, files in os.walk(directory_pos):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r', encoding='utf-8-sig') as f:
                    text = f.read()
                    pos_list.append([text, 0])

    for root, dirs, files in os.walk(directory_neg):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r', encoding='utf-8-sig') as f:
                    text = f.read()
                    neg_list.append([text, 1])


    return pos_list, neg_list