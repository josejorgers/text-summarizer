import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def verify(data):
    print('-------------------------------------------------------------------------')
    print('Text and summary length verification in order to keep really usefull data')
    print('-------------------------------------------------------------------------')
    print('')
    
    txtWordCount = []
    sumWordCount = []

    for t in data['text']:
        txtWordCount.append(len(t.split()))

    for s in data['summary']:
        sumWordCount.append(len(s.split()))

    df = pd.DataFrame({'text':txtWordCount, 'summary': sumWordCount})
    df.hist(bins=30)
    # plt.show()

    c = 0
    for s in data['summary']:
        if len(s.split()) <= 8:
            c += 1
    print('Proportion of summaries with 8 words or less:')
    print(c/len(data['summary']))

    c = 0
    for s in data['text']:
        if len(s.split()) <= 30:
            c += 1
    print('Proportion textssummaries with 8 words or less:')
    print(c/len(data['text']))


    max_summary, max_text = 8, 30
    short_summaries, short_texts = [], []
    
    text = np.array(data['text'])
    summary = np.array(data['summary'])

    for i in range(len(text)):
        if len(summary[i].split()) <= max_summary and len(text[i].split()) <= max_text:
            short_texts.append(text[i])
            short_summaries.append(summary[i])

    df = pd.DataFrame({'text': short_texts, 'summary': short_summaries})

    df['summary'] = df['summary'].apply(lambda x : 'beginsum ' + x + ' endsum')

    return df, max_text, max_summary
