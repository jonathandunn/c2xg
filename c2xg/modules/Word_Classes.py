import os
import time
import codecs
import random
import numpy as np
import multiprocessing as mp
from functools import partial
from sklearn import metrics
from gensim.models.word2vec import Word2Vec
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import BallTree

class Word_Classes(object):

	def __init__(self, Loader, use_pos = "POS"):
	
		self.language = Loader.language
		self.Encoder = Encoder(Loader = Loader, word_classes = True)
		self.Loader = Loader
		self.use_pos = use_pos
		
	#-------------------------------------------------------------------------------------------------#
	
	def train(self, size, min_count, sg = 1, hs = 1, iter = 20, max_vocab = None, workers = 1):

		#Generate filename for model
		model_name = str(self.language + "." + self.use_pos + "." + str(size) + "dim." + str(min_count) + ".min.Vectors.Incomplete.p")
		print("\tModel name: " + str(model_name))
		
		#Initialize sentence generator
		input_files = self.Loader.list_input()
		sentences = self.Encoder.load_stream(input_files, word_classes = True)
		
		#Check if model already exists
		if self.Loader.check_file(model_name) == True:
		
			model, iter_done = self.Loader.load_file(model_name)
			print("\tLoading model already trained for " + str(iter_done) + " iterations.")
			iter_full = iter
			iter = iter - iter_done
		
		#Need to initiate model
		else:
			print("\tInitializing word2vec model.")
			model = Word2Vec(size = size, 
								window = 5, 
								min_count = min_count, 
								workers = workers, 
								sg = sg, 
								hs = hs, 
								hashfxn = murmurhash3_32,
								sorted_vocab = 1,
								max_vocab_size = max_vocab,
								iter = 1
								)
			#Set iter counters
			iter = iter
			iter_done = 0
			iter_full = iter
		
			#Do initial training
			print("\tBuilding vocabulary.")
			model.build_vocab(sentences, update = False)
			
			print("\tInitial training cycle.")
			model.train(sentences, total_examples = model.corpus_count, epochs = 1)
			iter_done += 1
			
			print("\tSaving initial model.")
			self.Loader.save_file((model, iter_done), model_name)

		#If more training needed, keep training
		if iter_done < iter_full:
		
			for i in range(0, iter):
				print("\tStarting iteration " + str(iter_done + 1))
				#model.build_vocab(sentences, update = True)
				model.train(sentences, total_examples = model.corpus_count, epochs = 1)
				iter_done += 1
				
				print("\tSaving initial model.")
				self.Loader.save_file((model, iter_done), model_name)
				print("\tFinished training cycle.")
			
		#Now save and return
		model_name = model_name.replace("Incomplete", "Complete")
		self.Loader.save_file(model, model_name)
			
		return model
	#----------------------------------------------------------------------------------#
	
	def xmeans_clusters(self, word_vectors):
	
		#Initalize a k-means object and use it to extract centroids
		print("Starting K-Means++ initializer.")
		initial_centers = kmeans_plusplus_initializer(word_vectors, 5).initialize()
		
		print("Starting X-Means proper.")
		xmeans_instance = xmeans(word_vectors, initial_centers, kmax = 500, ccore = False)
		xmeans_instance.process()
		clusters = xmeans_instance.get_clusters()
			
		#Get the end time and print how long the process took
		end = time.time()
		elapsed = end - start
		print("Time taken for X Means clustering: " + str(elapsed) + " seconds.")
		
		return clusters
	#----------------------------------------------------------------------------------#	
	
	def agglomerative_clusters(self, word_vectors):
	
		#Pre-calculate BallTree object
		starting = time.time()
		Ball_Tree = BallTree(word_vectors, leaf_size = 200, metric = "minkowski")
		print("BallTree object in " + str(time.time() - starting))
		
		#Pre-calculate k_neighbors graph
		starting = time.time()
		connectivity_graph = kneighbors_graph(Ball_Tree, 
						n_neighbors = 1, 
						mode = "connectivity", 
						metric = "minkowski", 
						p = 2, 
						include_self = False, 
						n_jobs = workers
						)
		print("Pre-compute connectivity graph in " + str(time.time() - starting))

		#Agglomerative clustering
		starting = time.time()
		Agl = AgglomerativeClustering(n_clusters = 100, 
										affinity = "minkowski", 
										connectivity = connectivity_graph, 
										compute_full_tree = True, 
										linkage = "average"
										)
		
		Agl.fit(word_vectors)
		print("Agglomerative clustering in " + str(time.time() - starting))
		
		clusters = Agl.labels_
		
		return clusters
	#----------------------------------------------------------------------------------#
	
	def build_clusters(self, model_file, nickname, cluster_type = "xmeans", workers = 1):

		#Load and prep word embeddings
		if isinstance(model_file, str):
			print("Loading model")	
			model = gensim.models.Word2Vec.load(model_file)
			print("Done loading model")
				
		else:
			model = model_file

		start = time.time()

		print("Setting word vectors and number of clusters.")
		word_vectors = model.wv
		word_vectors = word_vectors.syn0
		print(word_vectors.shape)
		
		#Run relevant clustering algorithm
		if cluster_type == "xmeans":
			clusters = xmeans_clusters(word_vectors)
			
		elif cluster_type == "agglomerative":
			clusters = agglomerative_clusters(word_vectors, workers)		
		
		#Now convert clusters to word:cluster pairs
		cluster_dict = {}
		max_cluster = 0
				
		for i in range(len(clusters)):
			for word in clusters[i]:
						
				text = model.wv.index2word[word]
				word_list = text.split("/")
				pos = word_list[1]
						
				if str(i) + pos in cluster_dict:
					current_cluster = cluster_dict[str(i) + pos]
							
				else:
					cluster_dict[str(i) + pos] = max_cluster
					max_cluster += 1
							
				write_pos_dict[text] = cluster_dict[str(i) + pos]
					
		return write_dict, write_pos_dict
		
	#-------------------------------------------------------------------------------#
	#Write clusters in readable comma separated format -----------------------------#

	def write_clusters(self, input_clusters, nickname, output_file):

		with codecs.open(os.path.join(self.Loader.output_dir, nickname + "." + output_file + ".Clusters.txt"), "w", encoding = "utf-8") as fw:

			for key in list(set(sorted(input_clusters.values()))):
					
				for word in sorted(input_clusters.keys()):
					if input_clusters[word] == key:
							
						fw.write(word + "," + str(key) + "\n")
						
		self.Loader.save_file(input_clusters, nickname + "." + output_file + ".Dict.p")
	#-------------------------------------------------------------------------------#