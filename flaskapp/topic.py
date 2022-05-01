import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import gensim
import csv
import re
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

fields = ['abstract','title']
papers = pd.read_csv('../data/embedding_10000.csv',usecols=fields)
papers['abstract_title'] = papers['abstract'].astype(str) + papers['title'].astype(str)

# Print heads
#print(papers['abstract_title'].head())

# Remove punctuation
papers['abstract_title'] = papers['abstract_title'] .map(lambda x: re.sub('[,\.!?]', '', x))

# Convert the titles to lowercase
papers['abstract_title'] = papers['abstract_title'].map(lambda x: x.lower())

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use','NaN'])

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]
data = papers['abstract_title'].values.tolist()
data_words = list(sent_to_words(data))

# remove stop words
data_words = remove_stopwords(data_words)

print(len(data_words))

import gensim.corpora as corpora

# Create Dictionary
id2word = corpora.Dictionary(data_words)

# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
#print(corpus[:1][0][:30])

from pprint import pprint

# number of topics
num_topics = 50

# Build LDA model
lda_model = gensim.models.LdaModel(corpus=corpus,
                                   id2word=id2word,
                                   num_topics=num_topics)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())

#for i in range(len(corpus)):
   # print(i, lda_model.get_document_topics(corpus[i]))


doc_lda = lda_model[corpus]
import pyLDAvis.gensim_models as gensimvis
import pickle
import pyLDAvis

# Visualize the topics

visualisation = gensimvis.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(visualisation, 'LDA_Visualization10.html')


with open('trained_lda_10000.csv', 'w') as csvfile:
    csvfile.write(','.join(['id'] + [f'{i}' for i in range(num_topics)]) + '\n')

    for i in range(len(corpus)):
        newarray = ['0']*(num_topics + 1)
        newarray[0] = str(i)
        for x, y in lda_model.get_document_topics(corpus[i]):
            newarray[x+1] = str(y)
        csvfile.write(','.join(newarray) + '\n')

data = {}
for i in range(len(corpus)):
    data[i] = []
    for x, y in lda_model.get_document_topics(corpus[i]):
        data[i].append({
            x: str(y)
        })

with open('trained10000.json', 'w') as outfile:
    json.dump(data, outfile, indent=2)
