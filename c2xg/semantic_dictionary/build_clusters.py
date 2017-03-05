#----------------------------------------------------------------------------------#
#Load GenSim model and build clusters using K-Means -------------------------------#
#----------------------------------------------------------------------------------#
def build_clusters(model_file, 
					num_clusters, 
					output_file
					):

	import gensim
	from sklearn.cluster import KMeans
	import time
	import pickle
	import os

	#-------------------------------------------#
	def hash32(value):
		return hash(value) & 0xffffffff
	#-------------------------------------------#

	print("Loading model")	
	model = gensim.models.Word2Vec.load(model_file)
	print("Done loading model")

	start = time.time()

	print("Setting word vectors and number of clusters.")
	word_vectors = model.syn0

	#Initalize a k-means object and use it to extract centroids
	print("Starting K-Means clustering.")
	kmeans_clustering = KMeans(n_clusters = num_clusters)
	idx = kmeans_clustering.fit_predict(word_vectors)

	#Get the end time and print how long the process took
	end = time.time()
	elapsed = end - start
	
	print("Time taken for K Means clustering: " + str(elapsed) + " seconds.")

	#Create a Word / Index dictionary, mapping each vocabulary word to a cluster number#                                                                                           
	word_centroid_map = dict(zip(model.index2word, idx))

	#Save clusters#
	if os.path.isfile(output_file):
		os.remove(output_file)
		
	with open(output_file, 'wb') as f:
		pickle.dump(word_centroid_map, f)
	
	return
#-------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------#