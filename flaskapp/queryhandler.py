import flask
from transformers import AutoTokenizer, AutoModel
from csv import reader
import sklearn
import scipy
import pandas as pd
import os
import time
import flaskapp
import json
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

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


def getsimilarity(row,query_embedding):
    #print(row)
    vec1 = query_embedding.cpu().detach().numpy()
    vec2 = row.to_numpy()
    return 1 - scipy.spatial.distance.cosine(vec1, vec2)

def getrelateddocuments(query_title:str, query_abstract:str, cutoff = 0.8):
    
    os.chdir(os.path.dirname(__file__))
    embedding = pd.read_csv(flaskapp.embedding_file)
    #embedding = pd.read_csv('../data/embedding_10000.csv')
    embeddings = embedding.copy()

    query_embedding = getqueryembedding (query_title,query_abstract)

    embedding['similarity'] = embedding.apply(lambda row: getsimilarity(row[2:-2],query_embedding), axis=1)
    embedding = embedding.sort_values(by=['similarity'], ascending=False)
    embedding = embedding[["ID", "0", "title","abstract","similarity"]].head(10)
    embedding = embedding.rename(columns={'0': 'cord_uid'})
    embedding['abstract'] = embedding['abstract'].str[:300]
    getrelated = json.loads(embedding.to_json(orient="records"))
    
    IDs = [i['ID'] for i in getrelated]
    X= pd.DataFrame(columns = embeddings.columns[2:770])
    for i,j in enumerate(IDs):
        X = X.append(embeddings[embeddings['ID']==j].iloc[:,2:770])
    dist  = pdist(X, metric ='cosine')
    sim = 1-squareform(dist)
    sim[sim<cutoff]=0
    sim[sim>=cutoff]=1

    sim_df = pd.DataFrame(sim, columns = [i for i in IDs], index = [i for i in IDs])
    nodes_edges = {}
    for i in sim_df.index.to_list():
        f = sim_df.loc[i].to_list()
        nodes_edges[i] = [sim_df.columns[j] for j,k in enumerate(f) if (k ==1) & (i != sim_df.columns[j])]
    
    embedding['edges'] = embedding['ID'].map(nodes_edges)

    return   embedding.to_json(orient="records")

def test_getrelateddocuments():
    start = time.time()
    print(getrelateddocuments("Obesity and COVID-19", "Obesity and COVID-19"))
    end = time.time()
    print ("Total Execution time"+str(end-start))


def similarity_matrix(query_title:str, query_abstract:str,cutoff = 0.8):
    embedding,getrelated = getrelateddocuments(query_title, query_abstract)
    getrelated = json.loads(getrelated)
    IDs = [i['ID'] for i in getrelated]
    X= pd.DataFrame(columns = embedding.columns[2:770])
    for i,j in enumerate(IDs):
        X = X.append(embedding[embedding['ID']==j].iloc[:,2:770])
    dist  = pdist(X, metric ='cosine')
    sim = 1-squareform(dist)
    sim[sim<cutoff]=0
    sim[sim>=cutoff]=1
    return IDs,sim
            
def nodes_edges(query_title:str, query_abstract:str,cutoff=0.8):
    IDs,sm= similarity_matrix(query_title, query_abstract,cutoff)
    sm_df = pd.DataFrame(sm, columns = [i for i in IDs], index = [i for i in IDs])
    nodes_edges = {}
    for i in sm_df.index.to_list():
        f = sm_df.loc[i].to_list()
        nodes_edges[i] = [sm_df.columns[j] for j,k in enumerate(f) if (k ==1) & (i != sm_df.columns[j])]
    return nodes_edges















