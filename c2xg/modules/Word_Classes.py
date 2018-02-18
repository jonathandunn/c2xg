import os
import time
import codecs

from gensim.models.word2vec import Word2Vec
from modules.clustering.xmeans import XMeans
from modules.Encoder import Encoder

class Word_Classes(object):

	def __init__(self, language, Loader, use_pos = "POS"):
	
		self.language = language
		self.Encoder = Encoder(language = language, Loader = Loader, word_classes = True)
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

	def build_clusters(self, model_file, nickname):
	
		#Set input file as nickname
		self.nickname = nickname

		if isinstance(model_file, str):
			print("Loading model")	
			model = gensim.models.Word2Vec.load(model_file)
			print("Done loading model")
				
		else:
			model = model_file

		print("Setting word vectors and number of clusters.")
		word_vectors = model.wv
		word_vectors = word_vectors.syn0
		print(word_vectors.shape)

		#Perform x-means clustering
		starting = time.time()
		
		x = XMeans()
		x.fit(word_vectors)
		
		x_name = str(self.nickname + ".XMeans.p")
		self.Loader.save_file(x, x_name)
		
		print("\tCreated " + str(len(x.cluster_sizes_)) + " clusters in " + str(time.time() - starting) + " seconds.")
		
		clusters = x.labels_

		#Now create dictionaries of {words: cluster} pairs
		write_dict = {}
		write_pos_dict = {}
			
		#First, ignore POS tags
		for i in range(len(clusters)):
		
			word = model.wv.index2word[i]
			cluster = clusters[i]
			
			write_dict[word] = cluster
			
		#Second, enforce consistency of tags within clusters
		cluster_dict = {}
		max_cluster = 0
				
		for i in range(len(clusters)):

			word = model.wv.index2word[i]
			cluster = clusters[i]
			word_list = word.split("/")
			pos = word_list[1]
						
			if str(cluster) + pos in cluster_dict:
				current_cluster = cluster_dict[str(cluster) + pos]
							
			else:
				cluster_dict[str(cluster) + pos] = max_cluster
				max_cluster += 1
						
			write_pos_dict[word] = cluster_dict[str(cluster) + pos]
		
		return write_dict, write_pos_dict
		
	#-------------------------------------------------------------------------------#
	#Write clusters in readable comma separated format -----------------------------#

	def write_clusters(self, input_clusters, output_file):

		with codecs.open(os.path.join(self.Loader.output_dir, self.nickname + "." + output_file + ".Clusters.txt"), "w", encoding = "utf-8") as fw:

			for key in list(set(sorted(input_clusters.values()))):
					
				for word in sorted(input_clusters.keys()):
					if input_clusters[word] == key:
							
						fw.write(word + "," + str(key) + "\n")
						
		self.Loader.save_file(input_clusters, self.nickname + "." + output_file + ".Dict.p")
	#-------------------------------------------------------------------------------#