import pandas as pd
import numpy as np
import warnings
from preprocessing import text_preprocessing
from length_verification import verify
from sklearn.model_selection import train_test_split
from tokenizer import tokenization
from model import build_models, seqToText, seqToSummary, decodeSeq
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

data = pd.read_csv("./input/amazon-fine-food-reviews/Reviews.csv",nrows=100000)

data.drop_duplicates(subset = 'Text', inplace=True)
data.dropna(axis=0, inplace=True)


preprocessedText = []

for t in data['Text']:
    preprocessedText.append(text_preprocessing(t))

print('------------------------------')
print('Examples of preprocessed texts')
print('------------------------------')
print(preprocessedText[:5])
print('')


preprocessedSummary = []

for s in data['Summary']:
    preprocessedSummary.append(text_preprocessing(s, flag=True))


print('----------------------------------')
print('Examples of preprocessed summaries')
print('----------------------------------')
print(preprocessedSummary[:10])
print('')


data['text'] = preprocessedText
data['summary'] = preprocessedSummary

data.replace('', np.nan, inplace=True)
data.dropna(axis=0, inplace=True)

#### Analisys of the distribution of lengths of texts and summaries in order to select the maximum length...
#### for each one...

df, textLen, summaryLen = verify(data)

X_train, X_test, Y_train, Y_test = train_test_split(np.array(df['text']), np.array(df['summary']),
                                                    test_size=.1, random_state=0, shuffle=True)

X_train, X_test, Y_train, Y_test, X_voc, Y_voc, reverseXIndexWord, reverseYIndexWord, yWordIndex = tokenization(X_train,
                                                              X_test, Y_train, Y_test, textLen, summaryLen)


model, incModel, decModel = build_models(textLen, summaryLen, X_voc, Y_voc, X_train, Y_train, X_test, Y_test)



print('SOME EXAMPLES ON THE TRAINING DATA')
for sample in range(100):
    print('Review: ', seqToText(X_train[i], reverseXIndexWord))
    print('Original Summary: ', seqToSummary(Y_train[i], yWordIndex, reverseYIndexWord))
    print('Prediction: ', decodeSeq(X_train[i], model, encModel, decModel, reverseYIndexWord, yWordIndex, summaryLen))
    print('')
    print('')

print('SOME EXAMPLE ON THE TEST SET')
for sample in range(100):
    print('Review: ', seqToText(X_test[i], reverseXIndexWord))
    print('Original Summary: ', seqToSummary(Y_test[i], yWordIndex, reverseYIndexWord))
    print('Prediction: ', decodeSeq(X_test[i], model, encModel, decModel, reverseYIndexWord, yWordIndex, summaryLen))
    print('')
    print('')
