import json
import numpy as np
import warnings
import sys, os


targetWI = open(os.path.join(sys.argv[2], 'summary_word_index.json'))
target_word_index = json.load(targetWI)

reverseSourceWI = open(os.path.join(sys.argv[2], 'reverse_text_word_index.json'))
reverse_source_word_index = json.load(reverseSourceWI)

reverseTargetWI = open(os.path.join(sys.argv[2],'reverse_summary_word_index.json'))
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
    texts = np.loadtxt(os.path.join(sys.argv[2], 'textTest.csv'))
    summaries = np.loadtxt(os.path.join(sys.argv[2], 'summaryTest.csv'))

    import recurrent_model, convolutional_model
    import keras.backend as K
    K.clear_session()

    model = sys.argv[1]

    if model == 'conv':
        enc, dec = convolutional_model.getModel()
    elif model == 'rec':
        enc, dec = recurrent_model.getModel()
    elif model == 'lsa':
        import latent_semantic as lsa
        import pandas as pd

        predict = lsa.run()
        PATH = os.path.join(sys.argv[2], 'testDataset.csv')
        data = pd.read_csv(PATH)
        texts = data['text']
        sums = data['summary']
        
        for i in range(10):
            print("TEXT: " + str(i+1) + "\r\n")
            print("Review: " + texts[i] + "\r\n")
            print("Original summary: " + sums[i][7:-7] + "\r\n")
            print("Calculated words: \r\n")
            print(predict[2*i])
            print(predict[2*i + 1])
            print("\r\n")

        sys.exit(0)
    else:
        print('USAGE: python3 inference.py <model> <dataPath>')
        print('<model>: One of three options ("lsa", "conv" or "rec")')
        print('<dataPath>: Path to the test data')
        sys.exit(1)

    for i in range(0,500):
        print("TEXT: " + str(i+1) + "\r\n")
        print("Review: " + seq2text(texts[i]) + "\r\n")
        print("Original summary: " + seq2summary(summaries[i]) + "\r\n")
        print("Predicted summary: " + decode_sequence(texts[i].reshape(1,max_text_len), enc, dec) + "\r\n")
        print("\r\n")
    sys.stdout.flush()