from keras import backend as K
from attention import AttentionLayer
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

K.clear_session()

latent_dim = 300
embedding_dim = 100

def build_models(textLen, summaryLen, xVoc, yVoc, X_train, Y_train, X_test, Y_test):

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

    decEmbedLayer = Embedding(yVoc, embedding_dim, trainable=True)
    decEmbed = decEmbedLayer(decInput)

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


    model =  Model([encInput, decInput], decOut)

    ### TRAIN MODEL TO CREATE THE INFERENCE LATER ###

    model.summary()

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

    ## Early stopping to stop the training process once the loss function starts to increment
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)


    history = model.fit([X_train, Y_train[:,:-1]],
                    Y_train.reshape(Y_train.shape[0],
                                    Y_train.shape[1], 1)[:,:-1], epochs=20,
                    batch_size=128,
                    validation_data=([X_test, Y_test[:,:-1]],
                                     Y_test.reshape(Y_test.shape[0], Y_test.shape[1],
                                                    1)[:,:-1]))
    model.save('model-seq2seq-attn.h5')

    print('HISTORY OF THE MODEL')
    print(history)
    print('')
    print('')

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    
    ###################
    #### INFERENCE ####
    ###################


    # Input seq enconding to get the input vector
    encModel = Model(inputs=encInput, outputs=[encOut, stateH, stateC])

    ##### DECODER ####

    # Below tensors will hold the states of the previous time step
    decStateInputH = Input(shape=(latent_dim,))
    decStateInputC = Input(shape=(latent_dim,))
    decHiddenStateInput = Input(shape=(textLen, latent_dim))

    # Get the embeddings
    decEmbed2 = decEmbedLayer(decInput)

    # To predict the next word in the sequence, set the initial states
    # to the states from the previous time step
    decOut2, stateH2, stateC2 = decLSTM(decEmbed2, initial_state=[decStateInputH, decStateInputC])


    # Attention inference
    attOutInf, attStatesInf = attLayer([decHiddenStateInput, decOut2])
    decInfConcat = Concatenate(axis=-1, name='concat')([decOut2, attOutInf])

    #  A dense softmax layer to generate prob dist. over the target vocabulary
    decOut2 = decoder_dense(decoder_inf_concat) 

    # Final decoder model
    decModel = Model( [decoder_inputs] +
                           [decoder_hidden_state_input,decoder_state_input_h,
                            decoder_state_input_c], [decoder_outputs2] + [state_h2, state_c2])

    encModel.save('encModel-seq2seq-attn.h5')
    decModel.save('decModel-seq2seq-attn.h5')
    
    return model, encModel, decModel


def decodeSeq(inputSeq, model, encModel, decModel, reverseYIndexWord, yWordIndex, sumLen):

    # Seq encoding
    eOut, eH, eC = encModel.predict(inputSeq)

    targetSeq = np.zeros((1,1))
    targetSeq[0,0] = yWordIndex['beginsum']

    stop = False

    decodedStr = ''

    while not stop:
        tokensOut, h, c = decModel.predict([targetSeq] + [eOut, eH, eC])

        sampledTokIdx = np.argmax(tokensOut[0, -1, :])
        sampledTok = reverseYIndexWord[sampledTokIdx]

        if sampledTok != 'endsum':
            decodedStr += ' ' + sampledTok

        if sampledTok == 'endsum' or len(decodedStr.split()) >= (sumLen - 1):
            stop = True

        targetSeq = np.zeros((1,1))
        targetSeq[0,0] = sampledTokIdx

        eH, eC = h, c

    return decodedStr


def seqToSummary(inputSeq, yWordIndex, reverseYIndexWord):
    newStr = ''
    for i in inputSeq:
        if i != 0 and i != yWordIndex['beginsum'] and i != yWordIndex['endsum']:
            newString += reverseYIndexWord[i] + ' '
    return newStr

def seqToText(inputSeq, reverseXIndexWord):
    newStr = ''
    for i in inputSeq:
        if i != 0:
            newStr += reverseXIndexWord[i] + ' '
    return newStr
