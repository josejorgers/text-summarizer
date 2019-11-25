from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
import numpy as np

'''
A tokenizer builds a vocanulary and converts a word sequence into an integer sequence.

Here we make the tokenizer for every set of data
Also we define a rare word as a word which appear in the set less than a certain amount of times
Then we analize the % of rare words in the vocabulary and the amount of times that this words appear in texts
'''

def tokenize(train, test, textLen, tresh=4): #We define a rare word as a word that appear less than 'tresh' times

    tok = Tokenizer()
    tok.fit_on_texts(list(train))

    c = 0
    total_count = 0
    freq = 0
    total_freq = 0

    for key, value in tok.word_counts.items():
        total_count += 1
        total_freq += value
        if value < tresh:
            c += 1
            freq += value

    print('% of rare words in vocab:', (c/total_count) * 100.0)
    print('total coverage of rare words:', (freq/total_freq) * 100.0)

    tok = Tokenizer(num_words = total_count - c) # Training data with not rare words
    tok.fit_on_texts(list(train))

    # Convert from text into integer sequence
    train_seq = tok.texts_to_sequences(train)
    test_seq = tok.texts_to_sequences(test)

    # Padding zero where needed
    tr = pad_sequences(train_seq, maxlen=textLen,padding='post')
    te = pad_sequences(test_seq, maxlen=textLen,padding='post')

    voc = tok.num_words + 1
    print('Size of vocabulary:')
    print(voc) # + 1 cause the padding token

    return tr, te, voc 

def tokenization(X_train, X_test, Y_train, Y_test, xMaxLen, yMaxLen):

    X_tr, X_te, X_voc = tokenize(X_train, X_test, xMaxLen)
    Y_tr, Y_te, Y_voc = tokenize(Y_train, Y_test, yMaxLen, tresh=6)

    #Erase the empty summaries entries
    idx = []
    for i in range(len(Y_tr)):
        c = 0
        for j in Y_tr[i]:
            c += 1
        if c == 2:
            idx.append(i)
            
    Y_tr = np.delete(Y_tr, idx, axis=0)
    X_tr = np.delete(X_tr, idx, axis=0)

    idx = []
    for i in range(len(Y_te)):
        c = 0
        for j in Y_te[i]:
            c += 1
        if c == 2:
            idx.append(i)

    Y_te = np.delete(Y_te, idx, axis=0)
    X_te = np.delete(X_te, idx, axis=0)
    
    
    return X_tr, X_te, Y_tr, Y_te, X_voc, Y_voc
