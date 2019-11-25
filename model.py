from keras import backend as K
from attention import AttentionLayer
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model

K.clear_session()

latent_dim = 300
embedding_dim = 100

def build_model(textLen, summaryLen, xVoc, yVoc):

    ##### ENCODER #####

    encInput = Input(shape=(textLen,))

    encEmbed = Embedding(xVoc, embedding_dim, trainable=True)(encInput)

    encLSTM1 = LSTM(latent_dim, return_sequences=True,return_state=True,dropout=.4,recurrent_dropout=.4)
    encOut1, stateH1, stateC1 = encLSTM1(encEmbed)

    encLSTM2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=.4, recurrent_dropout=.4)
    encOut2, stateH2, stateC1 = encLSTM2(encOut1)

    encLSTM3 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=.4, recurrent_dropout=.4)
    encOut, stateH, stateC = encLSTM3(encOut2)

    ##### DECODER #####

    decInput = Input(shape=(None, ))

    decEmbed = Embedding(yVoc, embedding_dim, trainable=True)(decInput)

    decLSTM = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=.4, recurrent_dropout=.2)
    decOut, decFwState, decBackState = decLSTM(decEmbed, initial_state=[stateH, stateC])

    ##### ATTENTION #####

    attLayer = AttentionLayer(name='attention_layer')
    attOut, attStates = attLayer([encOut, decOut])

    ##### CONCAT ATTENTION INPUT AND DECODER LSTM OUTPUT #####

    decConcatInput = Concatenate(axis=-1, name='concat_layer')([decOut, attOut])

    ##### DENSE #####

    decDense = TimeDistributed(Dense(yVoc, activation='softmax'))
    decOut = decDense(decConcatInput)


    return Model([encInput, decInput], decOut)

    
