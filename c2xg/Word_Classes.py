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
    
from sklearn.metrics import pairwise_distances

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

        return model

    #----------------------------------------------------------------------------------#
    
    def learn_categories(self, model, vocab, unique_words = None, variety = "cbow", top_range = False):

        #Determine the number of clusters to use
        #Set clusters for syntactic domains
        if variety == "cbow":
            #If unspecified, set cbow max clusters
            if top_range == False:
                top_range = 250
            if top_range > len(vocab):
                top_range = len(vocab)-10
            cluster_range = range(top_range, 25, -5)

        #Set clusters for semantic domains
        elif variety == "sg":
            #If unspecified, set sg max clusters
            if top_range == False:
                top_range = 2500
            if top_range > len(vocab):
                top_range = len(vocab)-10
            cluster_range = range(top_range,250, -50)
            
        #Get the word embeddings specific to this lexicon
        #remove phrases because their vectors aren't trained
        word_list = [x for x in vocab.keys() if " " not in x]
        phrase_list = [x for x in vocab.keys() if " " in x]

        #Don't cluster very frequency words because they have their own behaviour
        word_list = [x for x in word_list if x not in unique_words.loc[:,"Word"].values]
        
        #Ensure not more clusters than words
        cluster_range = [x for x in list(cluster_range) if x < len(word_list)]
        
        #Get word vectors only for vocab
        vectors = []
        for word in word_list:
            vector = model.wv[word]
            vectors.append(vector)      
        vectors = np.vstack(vectors)

        print("Vocab and Vector Size:", end = "\t")
        print(len(word_list), vectors.shape)

        print("Getting cosine distance matrix")
        distances = pairwise_distances(X=vectors, Y=vectors, metric='cosine', n_jobs=mp.cpu_count())
        del vectors
        
        #Initialize search
        optimum_clusters = 0
        optimum_sh = 0.0
        n_turns_no_change = 0
        
        #Iterate over potential numbers of clusters
        for n_clusters in cluster_range:
            
            km = kmedoids.fasterpam(diss = distances, 
                                medoids = n_clusters, 
                                max_iter=100000, 
                                init='build', 
                                n_cpu=mp.cpu_count(),
                                )
            
            sh = kmedoids.medoid_silhouette(diss = distances, meds = km.labels)[0]
            print("\t", n_clusters, ": With ", km.n_iter, " iterations and ", km.n_swap, " swaps. Loss:", km.loss, " Silhoutette: ", sh)
            
            #Check for a better score
            if sh > optimum_sh + (sh * 0.001):
                print("\t\tBetter Silhoutette score obtained: ", optimum_sh, " now ", sh)
                optimum_sh = sh
                optimum_clusters = n_clusters
                n_turns_no_change = 0
                
                #Save current best clusters
                cluster_labels = km.labels
                cluster_exemplars = km.medoids
                
            else:
                n_turns_no_change += 1
                
            if n_turns_no_change > 2:
                print("No change for 3 iterations, stopping now.")
                break
        
        results = []
        for i in range(len(cluster_labels)):
            member_word = word_list[i]
            cluster = cluster_labels[i]
            exemplar_word = cluster_exemplars[cluster]
            exemplar_word_real = word_list[exemplar_word]
            results.append([member_word, cluster])
        
        cluster_df = pd.DataFrame(results, columns = ["Word", "Cluster"])

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
            mean_vector = model.wv.get_mean_vector(keys=ranks.loc[:,"Word"], weights=ranks.loc[:,"Rank"], pre_normalize=True, post_normalize=True)
            mean_dict[category] = mean_vector

            #Third, filter the best examples to reflect this particular lexicon
            exemplars = model.wv.similar_by_vector(mean_vector, topn=10000)
            exemplars = [x for x,y in exemplars if x in vocab.keys()]
            exemplars = [x for x in exemplars if x in word_list]

            #Make sure exemplars are relatively common words
            #mode_val = mode(list(vocab.values()))[0]
            #thresh_freq = mode_val * 3
            #exemplars = [x for x in exemplars if vocab[x] > thresh_freq]
            
            #If there are no exemplars, just use the first ones
            if len(exemplars) < 2:
                exemplars = category_df.loc[:,"Word"].values
            
            #Choose the top ones
            exemplars = exemplars[:2]
            name_dict[category] = exemplars
            exemplar_name = "_".join(exemplars)
            ranks.loc[:,"Category_Name"] = exemplar_name
            complete_clusters.append(ranks)

        #Assign phrases
        phrase_results = []

        for phrase in phrase_list:
            vector = model.wv[phrase]
            distances = pairwise_distances(vector.reshape(1, -1), list(mean_dict.values()), metric="cosine", n_jobs=mp.cpu_count())
            phrase_cluster = np.argmin(distances)
            phrase_cluster_name = "_".join(name_dict[phrase_cluster])
            phrase_results.append([0.0, phrase, phrase_cluster, phrase_cluster_name])

        #For CBOW, add unique_words as their own categories
        #For SG, unique_words do not belong to a cluster
        if variety == "cbow":
            unique_results = []
            starting = max(name_dict.keys()) + 1
            
            for word in unique_words.loc[:,"Word"].values:
                unique_results.append([0.0, word, starting, "unique_"+word])
                starting += 1
            unique_df = pd.DataFrame(unique_results, columns = ["Rank", "Word", "Category", "Category_Name"])
            complete_clusters.append(unique_df)

        #Merge and save all category rankings
        phrase_df = pd.DataFrame(phrase_results, columns = ["Rank", "Word", "Category", "Category_Name"])
        complete_clusters.append(phrase_df)
        
        #Merge, sort and prep the category lexicon
        cluster_df = pd.concat(complete_clusters)
        cluster_df = cluster_df.sort_values(by = ["Category", "Rank"], ascending=False)
        cluster_df = cluster_df.reset_index(drop=True)
  
        return cluster_df, mean_dict
        
    #-------------------------------------------------------------------------------#
    
    def learn_construction_categories(self, grammar, similarity_matrix):
    
        #Set range of construction clusters
        cluster_range = range(int(len(grammar)/5), 10, -10)
        
        #Initialize search
        optimum_clusters = 0
        optimum_sh = 0.0
        n_turns_no_change = 0
        
        #Iterate over potential numbers of clusters
        for n_clusters in cluster_range:
            
            km = kmedoids.fasterpam(diss = similarity_matrix, 
                                medoids = n_clusters, 
                                max_iter=100000, 
                                init='build', 
                                n_cpu=mp.cpu_count(),
                                )
            
            sh = kmedoids.medoid_silhouette(diss = similarity_matrix, meds = km.labels)[0]
            print("\t", n_clusters, ": With ", km.n_iter, " iterations and ", km.n_swap, " swaps. Loss:", km.loss, " Silhoutette: ", sh)
            
            #Check for a better score
            if sh > optimum_sh + (sh * 0.001):
                print("\t\tBetter Silhoutette score obtained: ", optimum_sh, " now ", sh)
                optimum_sh = sh
                optimum_clusters = n_clusters
                n_turns_no_change = 0
                
                #Save current best clusters
                cluster_labels = km.labels
                cluster_exemplars = km.medoids
                
            else:
                n_turns_no_change += 1
                
            if n_turns_no_change > 5:
                print("No change for 6 iterations, stopping now.")
                break
        
        results = []
        for i in range(len(cluster_labels)):
            member_construction = grammar[i]
            cluster = cluster_labels[i]
            results.append([member_construction, cluster])
        
        cluster_df = pd.DataFrame(results, columns = ["Chunk", "Cluster"])
    
        return cluster_df