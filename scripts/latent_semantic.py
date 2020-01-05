import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
import sys, os

PATH = os.path.join(sys.argv[2], 'testDataset.csv')

def divideInNGrams(text, n=2):
    result = []
    splitted = text.split()
    for t1 in range(len(splitted) - n):
        result += splitted[t1:t1+n]
    return result

def run():

    summaries = []
    texts = pd.read_csv(PATH)['text'][:10]
    for t in texts:
        doc = divideInNGrams(t)
        vectorizer = CountVectorizer()

        bag_of_words = vectorizer.fit_transform(doc)

        svd = TruncatedSVD(n_components=2)
        lsa = svd.fit_transform(bag_of_words)

        topic_encoded_df = pd.DataFrame(lsa, columns=['topic1','topic2'])
        topic_encoded_df['doc'] = doc

        dictionary = vectorizer.get_feature_names()

        encoding_matrix=pd.DataFrame(svd.components_,index=['topic1','topic2'],columns=dictionary).T

        encoding_matrix['abs_topic1']=np.abs(encoding_matrix["topic1"])
        encoding_matrix['abs_topic2']=np.abs(encoding_matrix["topic2"])
        
        final_matrix=encoding_matrix.sort_values('abs_topic1',ascending=False)
        final_matrix[['topic1','topic2']]

        sentence1= final_matrix[final_matrix["abs_topic1"]>=.5]
        summaries.append(sentence1[['abs_topic1']])

        sentence2=final_matrix[final_matrix["abs_topic2"]>=0.5]
        summaries.append(sentence2[['abs_topic2']])
    return summaries

