import os
import time
import codecs
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial

from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model
from gensim.models.fasttext import save_facebook_model
import kmedoids

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except:
    print("UNABLE TO ACCELERATE SKLEARN")

from sklearn import metrics
from sklearn.metrics.pairwise import cosine_distances

class Word_Classes(object):

    def __init__(self, Load):
    
        self.language = Load.language
        self.Load = Load

    #----------------------------------------------------------------------------------#
    def learn_embeddings(self, input_data, model_type, name="embeddings"):

        if model_type == "cbow":
            ws = 1
            sg = 0

        elif model_type == "sg":
            ws = 5
            sg = 1

        name = name+"."+self.language

        #Pre-load text
        data = [x for x in self.Load.load(input_data)]

        model = FastText(vector_size=100, window=ws, sg=sg, hs=0, negative=100, sorted_vocab=1, alpha=0.01, min_count=5, min_n=3, max_n=6)
        model.build_vocab(corpus_iterable=data)
        model.train(corpus_iterable=data, total_examples=len(data), epochs=20)

        #Save
        save_name = os.path.join(self.Load.out_dir, name+"."+model_type+".bin")
        print("Saving " + save_name)
        save_facebook_model(model, save_name)

        return

    #----------------------------------------------------------------------------------#
    
    def learn_categories(self, model_file, vocab):

        #Load and prep word embeddings
        if isinstance(model_file, str):
            print("Loading model")    
            model = load_facebook_model(model_file)
            print("Done loading model")
                
        else:
            model = model_file

        vectors = []
        for word in vocab.keys():
            vector = model.wv[word]
            vectors.append(vector)
                
        vectors = np.vstack(vectors)

        print("Vocab and Vector Size")
        print(len(vocab))
        print(vectors.shape)

        print("Getting cosine distance matrix")
        distances = cosine_distances(vectors, vectors)
        print(distances)

        km = kmedoids.KMedoids(5, method='fasterpam', max_iter = 1000000, init = "build")
        km.fit(distances)
        print("Loss is:", km.inertia_)

        cluster_labels = km.labels_
        cluster_exemplars = km.medoid_indices_
        print(cluster_labels)
        print(cluster_exemplars)

        results = []
        for i in range(len(cluster_labels)):
            member_word = list(vocab.keys())[i]
            cluster = cluster_labels[i]
            exemplar_word = cluster_exemplars[cluster]
            exemplar_word_real = list(vocab.keys())[exemplar_word]
            results.append([member_word, cluster, exemplar_word_real])
        
        cluster_df = pd.DataFrame(results)
        print(cluster_df)
                 
        return cluster_df
        
    #-------------------------------------------------------------------------------#