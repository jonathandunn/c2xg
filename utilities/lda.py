#Streaming LDA with GenSim to pandas DataFrame
#-- Jonathan Dunn, 3/9/2017
#-- jonathan.e.dunn.ctr@nga.mil

#This script learns an LDA model from all documents in the input directory.
#-- The LDA model provides a set of topics representing the content of these documents.
#-- The script creates a DataFrame with document-by-document topic info (assumes lines are documents)

import os
import codecs
import gensim

#Creates iterator for sending corpus to GenSim for making feature dictionary -------------------------#	
#-----------------------------------------------------------------------------------------------------#
class DictIterator(object):
	
	def __init__(self, dirname):
		self.dirname = dirname
		
	def __iter__(self):

		for fname in os.listdir(self.dirname):
			for line in codecs.open(os.path.join(self.dirname, fname), encoding = "utf-8"):
				
				line = gensim.utils.simple_preprocess(line, deacc = False, min_len = 2, max_len = 1000000)
				
				yield line
#---------------------------------------------------------------------------------------------------#
#Creates iterator for sending corpus to GenSim for making vectors ----------------------------------#
#---------------------------------------------------------------------------------------------------#
class CorpusStream(object):
	
	def __init__(self, dirname, dictionary):
		self.dirname = dirname
		self.dictionary = dictionary
		
	def __iter__(self):

		for fname in os.listdir(self.dirname):
			for line in codecs.open(os.path.join(self.dirname, fname), encoding = "utf-8"):
				
				line = gensim.utils.simple_preprocess(line, deacc = False, min_len = 2, max_len = 1000000)
				
				yield self.dictionary.doc2bow(line)
#---------------------------------------------------------------------------------------------------#
def train_topic_model(input_folder, nickname, topics, cpus):
	
	print("\tLearn cbow features from text documents in " + input_folder + ".")
	corpus = DictIterator(input_folder)
	corpus_dictionary = gensim.corpora.Dictionary(corpus)
	corpus_dictionary.save(nickname + ".Features")
		
	#Create iterator for extracting vectors from corpus	
	corpus_vectors = CorpusStream(input_folder, corpus_dictionary)
		
	print("\tLearn TF-IDF weights from text documents in " + input_folder + ".")
	tfidf = gensim.models.TfidfModel(corpus_vectors)
	tfidf.save(nickname + ".TF-IDF")
	
	#Update iterator with TF-IDF weights
	corpus_vectors_tfidf = tfidf[corpus_vectors]
		
	print("\tLearn LDA model streaming over TF-IDF vectors from " + input_folder + ".")
	lda_object = gensim.models.ldamulticore.LdaMulticore(corpus_vectors_tfidf, num_topics = topics, workers = cpus)
	lda_object.save(nickname + ".LDA")
	
	return corpus_dictionary, tfidf, lda_object
#---------------------------------------------------------------------------------------------------#
def run_topic_model(input_folder, nickname, corpus_dictionary = "", tfidf = "", lda_object = "", cpus = 1):
	
	#Annotate files in input folder and return as pandas DataFrame	
	import pandas as pd
	
	#If models are passed as strings, load via nickname as filename
	if corpus_dictionary == "":
		corpus_dictionary = gensim.corpora.Dictionary.load(nickname + ".Features")
		
	if tfidf == "":
		tfidf = gensim.models.TfidfModel.load(nickname + ".TF-IDF")
	
	if lda_object == "":
		lda_object = gensim.models.ldamulticore.LdaMulticore.load(nickname + ".LDA")
	
	#Create iterator for extracting vectors from corpus	
	corpus_vectors = CorpusStream(input_folder, corpus_dictionary)
	
	#Update iterator with TF-IDF weights
	corpus_vectors_tfidf = tfidf[corpus_vectors]
	
	print("\tEnriching data in " + input_folder + " and saving to pandas DataFrame")
	
	instance_list = []
	[instance_list.append([x[0][0], x[0][1]]) for x in lda_object[corpus_vectors_tfidf]]
	
	result_df = pd.DataFrame(instance_list, columns = ["Primary Topic", "Topic Fit"])
		
	return result_df
#---------------------------------------------------------------------------------------------------#
def read_topic_models(corpus_dictionary, lda_object, nickname):

	import codecs
	
	print("Writing topics to " + nickname + ".Topics.txt")
	fw = codecs.open(nickname + ".Topics.txt", "w", encoding = "utf-8")
	
	for topic in lda_object.show_topics(num_topics = -1):
		
		topic_id = topic[0]
		features = topic[1].split(" + ")
		
		fw.write("Topic " + str(topic_id) + ": ")
		
		for feature in features:
		
			feature = feature.split("*")
			loading = feature[0]
			word = int(feature[1])
			
			fw.write(str(corpus_dictionary.get(word)) + " ")
		
		fw.write("\n")
		
	fw.close()
	
	return
#---------------------------------------------------------------------------------------------------#
if __name__ == "__main__":

	input_folder = "./data"			#The input directory. All files will be streamed
	nickname = "Testing"			#Nickname used for saving results
	topics = 50						#Number of topic clusters
	cpus = 12						#Number of CPUs available for multi-processing

	#This is how the learning algorithm is called. Models are also saved to disk
	corpus_dictionary, tfidf, lda_object = train_topic_model(input_folder, nickname, topics, cpus)
	
	#This is how to write the topics to a readable file (avoids encoding issues with stdout)
	read_topic_models(corpus_dictionary, lda_object, nickname)
	
	#This is how to get a pandas dataframe with document-by-document topic information
	result_df = run_topic_model(input_folder, nickname, corpus_dictionary, tfidf, lda_object, cpus)
	
	print(result_df)