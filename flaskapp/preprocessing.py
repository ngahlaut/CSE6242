import numpy as np
import math
import pandas as pd
import scipy.io as spio
import sklearn.preprocessing as skpp
import scipy.sparse.linalg as ll
from os.path import abspath, exists
import os
import time
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

def BuildCovarianceMatrix(y):
    m,n = y.shape
    mu = np.mean(y,axis = 1)
    xc = y - mu[:,None]
    C = np.dot(xc,xc.T)/m
    return C,xc

def EigenDecomposition(K,C):
    S,W = ll.eigs(C,k = K)
    S = S.real
    W = W.real
    W = W.T
    return S,W

def GenerateEmbeddings(MetadataFile:str,MaxRows:int,OutFile:str):
    os.chdir(os.path.dirname(__file__))
    if os.path.exists("../data/"+OutFile):
        os.remove("../data/"+OutFile)
    column_names= [str(i) for i in range(1,769)]
    meta = pd.read_csv("../data/"+MetadataFile)
    len_meta = len(meta)
    start = np.random.randint(0,len_meta)
    counter=1
    while counter<=MaxRows:
        df = pd.DataFrame(columns = column_names)
        temp = meta.iloc[start:start+1,:]
        temp = temp.fillna('')
        #print(temp['title'].head(5))
        for index, row in temp.iterrows():
            #print(index)
            embedding = getqueryembedding(row['title'], row['abstract']).cpu().detach().numpy()
            df = df.append(pd.DataFrame(embedding.reshape(1,-1), columns=list(df)), ignore_index=True)
            df.insert(len(column_names), 'abstract', row['abstract'])
            df.insert(len(column_names)+1, 'title', row['title'])
            df.insert(0, '0', row['cord_uid'])
            df.insert(0, 'ID', counter)

        #print(df.head(5))
        if counter ==1:
            df.to_csv("../data/"+OutFile+".csv", sep=",",index=False,mode='a',header=True)
        else:
            df.to_csv("../data/"+OutFile+".csv", sep=",",index=False,mode='a',header=False)
        counter += 1
        start = np.random.randint(0,len_meta)

    print(pd.read_csv("../data/"+OutFile).head(10))



def GeneratePrincipalComponents(SourceData:str,OutFile:str):
    start = time.time()

    os.chdir(os.path.dirname(__file__))

    embedding = pd.read_csv("../data/"+SourceData)
    embedding_nocordid = embedding.iloc[:,2:770 ]
    A1 =embedding_nocordid.to_numpy()
    A1 = A1.T

    C1,xc1 = BuildCovarianceMatrix(A1)
    S1,W1 = EigenDecomposition(2,C1)
    principal_component1 = np.dot(W1.T[:,0],xc1)/math.sqrt(S1[0])
    principal_component2 = np.dot(W1.T[:,1],xc1)/math.sqrt(S1[1])
    index = np.arange(len(principal_component1))
    Principal_components = np.column_stack((index,principal_component1,principal_component2))

    df = pd.DataFrame(Principal_components, columns=['index','principal_component1','principal_component2'])
    df.to_csv('../data/'+OutFile, sep=",",index=False)
    df.to_csv('static/'+OutFile, sep=",",index=False)

    end = time.time()
    print ("Total Execution time"+str(end-start))


GenerateEmbeddings('metadata.csv',10000, 'embedding_10000')
GeneratePrincipalComponents('embedding_10000.csv', 'principal_components_10k.csv')


