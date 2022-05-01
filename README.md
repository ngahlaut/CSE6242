# COVID-19 Knowledge search and visualization 

[**Project Description**](#Project-Description)|[**Pretrained models**](#Pretrained-models) | [**Data PreProcessing**](#Data-PreProcessing) | [**Installation**](#Installation) 

# Project Description

Worldwide efforts on fighting COVID-19 has resulted in rapid increase of COVID-19 related publications. Efficiently and precisely identifying the relevant information for a specific research topic becomes challenging. To tackle this challenge, we developed an intelligent dashboard based on state-of-the-art Transformers model embedding (ref: SPECTER) associated with the latent dirichlet allocation (ref: LDA). The dataset we use is the COVID-19 Research literature dataset.(Lu Wang, Lo et al. 2020). To power the dashboard we built a flask application to query COVID-19 related publications and visualization that is built using Javascript.

# Pretrained models

## SPECTER

Here is an example:
```
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')
def getqueryembedding(query: str, abstract: str=''):
    title_abs = [query + tokenizer.sep_token + abstract ]
    # preprocess the input
    inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    result = model(**inputs)
    # take the first token in the batch as the embedding
    embeddings = result.last_hidden_state[:, 0, :]
    return embeddings

    embedding = getqueryembedding('Title', 'abstract')
```

## LDA
We collect the abstract and title for each paper in the dataset and build topic models by LDA. There are several steps involved:
1. Collect the data from csv
2. Remove punctuations
3. Convert sentences to words
4. Remove stop words
5. Feed the data into pyLDAvis to train the model
6. Get weights for each document in each topic
7. Output the result in csv format

Below is sample code for running LDA:
```
# Create Dictionary
id2word = corpora.Dictionary(data_words)

# Create Corpus
corpus = [id2word.doc2bow(text) for text in data_words]

# number of topics
num_topics = 50

# Build LDA model
lda_model = gensim.models.LdaModel(corpus=corpus,
                                   id2word=id2word,
                                   num_topics=num_topics)

# Get document topics
for x, y in lda_model.get_document_topics(corpus[i]):
    weights[x] = y
```


# Data PreProcessing

To support query search on covid19 literature we need to pre train a model and generate document embedding using transformer model. This is done to facilitate query search and computing similarity score. We use HuggingFace's transformers library to pre train model and generate document embedding.  

Document embedding generated using HuggingFace's transformers library produces a feature vector that consists of 768 features. Visualizing documents on a multi-dimensional feature space is difficult. To solve this problem, we generate top 2 principal components using PCA. This technique helps us in visualizing the similarity between the documents. 

The input to dataprocessing is a metadata.csv file that contains title and abstract for the scientific literature on Covid19. Example can be found here (https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge?select=metadata.csv)

The pre processed data can be found under data and flaskapp/static folders. 

To regenerate new embedding with the latest metadata.csv. Re run the preprocessing step with code located in flaskapp/preprocessing.py before starting the flask application

# Installation

In order to run the application on your machine follow the steps below

* Enable git large file storage before you clone the repo. Instructions for this can be found here (https://git-lfs.github.com/)
* Clone the repo locally
* Requires python 3.8. 
* Install required libraries from requirements.txt ```pip3 install -r requirements.txt ```
* Make sure no other application is running on port 8080
* Run ```python3 run.py ```

Once flask is up and running navigate to http://127.0.0.1:8080 (do not use http://localhost:8080). This should open an interactive dashboard with a default query search related to COVID-19. You can change the query string and watch the dashboard update the visualization based on your search parameters.

There is also swagger documentation for the API's which can be accessed from (http://127.0.0.1:8080/swagger)

If you see errors with installing transformers on your machine follow instructions below

* curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
* Restart the terminal
* Pip3 install transformers==4.12.5

# Demo video

The installation demo video can be found in link below https://www.youtube.com/watch?v=HigwjmwJgzQ

# Github repo
The github repo can be cloned (not downloaded) from the link here https://github.com/maniraja1/CSE6242
P.S: Do not download the repo but clone the repo as there are large files in the repo that may not be downloaded correctly
