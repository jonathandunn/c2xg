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
from scipy.stats import mode
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

        #Get the word embeddings specific to this lexicon
        #remove phrases because their vectors aren't trained
        word_list = [x for x in vocab.keys() if " " not in x]
        phrase_list = [x for x in vocab.keys() if " " in x]

        vectors = []
        for word in word_list:
            vector = model.wv[word]
            vectors.append(vector)
                
        vectors = np.vstack(vectors)

        print("Vocab and Vector Size:", end = "\t")
        print(len(word_list), vectors.shape)

        print("Getting cosine distance matrix")
        distances = cosine_distances(vectors, vectors)

        print("Clustering")
        km = kmedoids.KMedoids(5, method='fasterpam', max_iter = 1000000, init = "build")
        km.fit(distances)
        print("Loss is:", km.inertia_)

        cluster_labels = km.labels_
        cluster_exemplars = km.medoid_indices_
        #print(cluster_labels)
        #print(cluster_exemplars)

        results = []
        for i in range(len(cluster_labels)):
            member_word = word_list[i]
            cluster = cluster_labels[i]
            exemplar_word = cluster_exemplars[cluster]
            exemplar_word_real = word_list[exemplar_word]
            results.append([member_word, cluster])
        
        cluster_df = pd.DataFrame(results, columns = ["Word", "Cluster"])
        #print(cluster_df)

        #Save mean vectors for assigning phrases to clusters
        mean_dict = {}
        name_dict = {}
        complete_clusters = []

        #Get proto-type structure and exemplars
        for category, category_df in cluster_df.groupby("Cluster"):

            #First, get proto-type structure of the cluster
            ranks = model.wv.rank_by_centrality(words=category_df.loc[:,"Word"], use_norm=True)
            ranks = pd.DataFrame(ranks, columns = ["Rank", "Word"])
            ranks.loc[:,"Category"] = category
            

            #Second, get the best example of the category
            mean_vector = model.wv.get_mean_vector(keys=ranks.loc[:,"Word"], weights=ranks.loc[:,"Rank"], pre_normalize=True, post_normalize=False)
            mean_dict[category] = mean_vector

            #Third, filter the best examples to reflect this particular lexicon
            exemplars = model.wv.similar_by_vector(mean_vector, topn=10000)
            exemplars = [x for x,y in exemplars if x in vocab.keys()]
            mode_val = mode(list(vocab.values()))[0]
            thresh_freq = mode_val + 1
            #print("Threshold freq", thresh_freq)

            #Make sure exemplars are relatively common words
            exemplars = [x for x in exemplars if vocab[x] > thresh_freq]
            exemplars = exemplars[:4]
            name_dict[category] = exemplars
            exemplar_name = "_".join(exemplars)
            ranks.loc[:,"Category_Name"] = exemplar_name
            complete_clusters.append(ranks)

        #Assign phrases
        phrase_results = []

        for phrase in phrase_list:
            vector = model.wv[phrase]
            distances = cosine_distances(vector.reshape(1, -1), list(mean_dict.values()))
            phrase_cluster = np.argmin(distances)
            phrase_cluster_name = "_".join(name_dict[phrase_cluster])
            phrase_results.append([0.0, phrase, phrase_cluster, phrase_cluster_name])
            
        #Merge and save all category rankings
        phrase_df = pd.DataFrame(phrase_results, columns = ["Rank", "Word", "Category", "Category_Name"])
        complete_clusters.append(phrase_df)

        cluster_df = pd.concat(complete_clusters)
        cluster_df = cluster_df.sort_values(by = ["Category", "Rank"])
        cluster_df = cluster_df.reset_index(drop=True)
  
        return cluster_df
        
    #-------------------------------------------------------------------------------#