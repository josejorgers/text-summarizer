import pandas as pd
import numpy as np
import warnings
from preprocessing import text_preprocessing
from length_verification import verify
from sklearn.model_selection import train_test_split
from tokenizer import tokenization
from model import build_model
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

X_train, X_test, Y_train, Y_test, X_voc, Y_voc = tokenization(X_train,
                                                              X_test, Y_train, Y_test, textLen, summaryLen)


model = build_model(textLen, summaryLen, X_voc, Y_voc)

model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

## Early stopping to stop the training process once the loss function starts to increment
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)


history = model.fit([X_train, Y_train[:,:-1]],
                    Y_train.reshape(Y_train.shape[0],
                                    Y_train.shape[1], 1)[:,:-1], epochs=20,
                    callbacks=[es], batch_size=128,
                    validation_data=([X_test, Y_test[:,:-1]],
                                     Y_test.reshape(Y_test.shape[0], Y_test.shape[1],
                                                    1)[:,:-1]))
model.save('model-seq2seq-attn.h5')


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
