import os
import time
import codecs
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from sklearn import metrics
from sklearn.cluster import KMeans
from gensim.models import fasttext
from gensim.test.utils import datapath
from ..modules.Encoder import Encoder

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except:
    print("UNABLE TO ACCELERATE SKLEARN")

class Word_Classes(object):

    def __init__(self, Loader, use_pos = "POS"):
    
        self.language = Loader.language
        self.Loader = Loader
        self.Encoder = Encoder(Loader = Loader, word_classes = True)
        self.use_pos = use_pos
        
    #----------------------------------------------------------------------------------#
    
    def build_clusters(self, model_file, vocab, nickname):

        #Load and prep word embeddings
        if isinstance(model_file, str):
            print("Loading model")    
            model = fasttext.load_facebook_vectors(model_file)
            print("Done loading model")
                
        else:
            model = model_file

        vectors = []
        for word in vocab:
            if word != "&":
                vector = model.get_vector(word, norm=True)
                vectors.append(vector)
                
        vectors = np.vstack(vectors)

        print("Vocab and Vector Size")
        print(len(vocab))
        print(vectors.shape)
        
        kmeans = KMeans(n_clusters=int(len(vocab)/100), 
                            init='k-means++', 
                            n_init=1000, 
                            max_iter=300000, 
                            tol=0.0001, 
                            algorithm='auto'
                            )
                            
        print("Fitting k-means")
        kmeans.fit(vectors)
        
        #Now convert clusters to word:cluster pairs
        cluster_dict = {}
        max_cluster = 0
        clusters = kmeans.labels_
        output = []
                
        for i in range(len(clusters)):
            word = vocab[i]
            cluster = clusters[i]
            output.append([word, cluster])
                            
        df = pd.DataFrame(output)
         
        return df
        
    #-------------------------------------------------------------------------------#