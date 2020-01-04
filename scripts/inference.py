import json
import numpy as np
import warnings

targetWI = open('summary_word_index.json')
target_word_index = json.load(targetWI)

reverseSourceWI = open('reverse_text_word_index.json')
reverse_source_word_index = json.load(reverseSourceWI)

reverseTargetWI = open('reverse_summary_word_index.json')
reverse_target_word_index = json.load(reverseTargetWI)

max_text_len = 30
max_summary_len = 8

def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
            newString=newString+reverse_target_word_index[str(int(i))]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[str(int(i))]+' '
    return newString


def decode_sequence(input_seq, encoder_model, decoder_model):

    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    target_seq = np.zeros((1,1))
    
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[str(sampled_token_index)]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
            stop_condition = True

        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        e_h, e_c = h, c

    return decoded_sentence


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    texts = np.loadtxt('textTest.csv')
    summaries = np.loadtxt('summaryTest.csv')

    from recurrent_model import getModel
    import keras.backend as K
    K.clear_session()

    enc, dec = getModel()

    enc.summary()
    dec.summary()

    print('GOT ENCODER AND DECODER')
    print(len(target_word_index.keys()))
    print(len(reverse_source_word_index.keys()))
    for i in range(0,500):
        print("Review:",seq2text(texts[i]))
        print("Original summary:",seq2summary(summaries[i]))
        print("Predicted summary:",decode_sequence(texts[i].reshape(1,max_text_len), enc, dec))
        print("\n")
    print('END')