import os
import pickle
import codecs
import time
import re
import boto3
import os.path
import cytoolz as ct
from functools import partial
import multiprocessing as mp
from sklearn.cluster import KMeans
from sklearn.utils import murmurhash3_32
from gensim.parsing import preprocessing
from gensim.models.word2vec import Word2Vec

#-------------------------------------------------------------------------------------------------
def train_model(workers, sg, size, min_count, hs, iter, max_vocab, language, use_pos, vector_write, s3_bucket, s3_prefix):

	#Initialize RDRPosTagger
	model_string = os.path.join(".", "data", "pos_rdr", language + ".RDR")
	dict_string = os.path.join(".", "data", "pos_rdr", language + ".DICT")
	
	from modules.rdrpos_tagger.Utility.Utils import readDictionary
	from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger
	from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import unwrap_self_RDRPOSTagger
	from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import printHelp
	
	current_dir = os.getcwd()
	if current_dir == "Utility":
		os.chdir(os.path.join("..", "..", ".."))
	
	#Initialize tagger
	r = RDRPOSTagger()
	r.constructSCRDRtreeFromRDRfile(model_string) 
	DICT = readDictionary(dict_string) 
	tag_it = partial(r.tagRawSentenceOriginal, DICT = DICT)
	
	#Initialize data generator
	sentences = MySentences(tag_it, use_pos, workers, s3_bucket, s3_prefix, language)
	
	#Check if model already exists
	client = boto3.client("s3")
		
	response = client.list_objects_v2(
					Bucket = s3_bucket,
					Delimiter = "/",
					Prefix = "Dict_Models/"
					)
	
	print("\tCurrent model file: " + vector_write)
	files = []
	for key in response["Contents"]:
		files.append(key["Key"])
		
	if vector_write in files:	
		
		client.download_file(s3_bucket, vector_write, "temp.p")
		model, iter_done = read_file("temp.p")
		print("\tLoading model already trained for " + str(iter_done) + " iterations.")
		iter_full = iter
		iter = iter - iter_done
	
	
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
		write_file("temp.p", (model, iter_done))
		client.upload_file("temp.p", s3_bucket, vector_write)

	#If more training needed, keep training
	if iter_done < iter_full:
	
		for i in range(0, iter):
			print("\tStarting iteration " + str(iter_done + 1))
			#model.build_vocab(sentences, update = True)
			model.train(sentences, total_examples = model.corpus_count, epochs = 1)
			iter_done += 1
			
			write_file("temp.p", (model, iter_done))
			client.upload_file("temp.p", s3_bucket, vector_write)
			print("\tFinished training cycle.")
		
	#Now save and return
	write_file("temp.p", model)
	client.upload_file("temp.p", s3_bucket, vector_write + ".Finished.p")
		
	return model
#----------------------------------------------------------------------------------#
# Fixes integeter length for interfacing Python and C -----------------------------#

def hash32(value):
	return hash(value) & 0xffffffff
#----------------------------------------------------------------------------------#

def tokenize_zho(line, jb):

	line = [x for x in jb.cut(line, cut_all = True, HMM = True) if x != ""]
	line = " ".join(line)

	return line
#----------------------------

def format_line(line, myre, tag_it, use_pos):

	#Remove links, hashtags, at-mentions, mark-up, and "RT"
	line = re.sub(r"http\S+", "", line)
	line = re.sub(r"@\S+", "", line)
	line = re.sub(r"#\S+", "", line)
	line = re.sub("<[^>]*>", "", line)
	line = line.replace(" RT", "").replace("RT ", "")
				
	#Remove emojis
	line = re.sub(myre, "", line)
				
	#Remove punctuation and extra spaces
	line = ct.pipe(line, preprocessing.strip_tags, preprocessing.strip_punctuation, preprocessing.strip_numeric, preprocessing.strip_non_alphanum, preprocessing.strip_multiple_whitespaces)
				
	#Strip and reduce to max training length
	line = line.lower().strip().lstrip()
								
	if use_pos == True:
		line = tag_it(rawLine = line)
					
	#Now split into a list of words
	line = line.split(" ")

	return line
#-------------------------------------------------------------------------------------------#
#Creates iterator for sending to word2vec model, based on GenSim example ----------#

class MySentences(object):

	def __init__(self, tag_it, use_pos, workers, s3_bucket, s3_prefix, language):
		self.tag_it = tag_it	
		self.use_pos = use_pos
		self.workers = workers
		self.s3_bucket = s3_bucket
		self.s3_prefix = s3_prefix
		self.language = language
				
	def __iter__(self):
	
		try:
		# Wide UCS-4 build
			myre = re.compile(u'['
				u'\U0001F300-\U0001F64F'
				u'\U0001F680-\U0001F6FF'
				u'\u2600-\u26FF\u2700-\u27BF]+', 
				re.UNICODE)
		except re.error:
			# Narrow UCS-2 build
				myre = re.compile(u'('
				u'\ud83c[\udf00-\udfff]|'
				u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
				u'[\u2600-\u26FF\u2700-\u27BF])+', 
				re.UNICODE)
	
		#zho needs an additional tokenizer
		if self.language == "zho":
			
			import modules.jieba.jeiba as jb
			tk = jb.Tokenizer()
			tk.initialize()
			tk.lock = True
					
		#Data is chunked into a large number of files
		#Multi-process by loading each file into memory, distributing analysis, and yielding from memory
		
		#Get list of files in s3 bucket
		client = boto3.client("s3")
		
		response = client.list_objects_v2(
						Bucket = self.s3_bucket,
						Delimiter = "/",
						Prefix = self.s3_prefix + "/"
						)
		
		files = []
		for key in response["Contents"]:
			files.append(key["Key"])
		
		for fname in files:
			
			try:
				#Download file
				starting = time.time()
				client.download_file(s3_bucket, fname, "temp.txt")
				
				#Load this file into memory
				with codecs.open("temp.txt", encoding = "utf-8", errors = "replace") as fo:
					lines = fo.readlines()
					
				#Tokenize zho
				if self.language == "zho":
					
					pool_instance=mp.Pool(processes = self.workers, maxtasksperchild = None)
					lines = pool_instance.map(partial(tokenize_zho, jb = tk), lines, chunksize = 2000)
					pool_instance.close()
					pool_instance.join()				
					
				#Multi-process annotations and store in memory
				pool_instance=mp.Pool(processes = self.workers, maxtasksperchild = None)
				lines = pool_instance.map(partial(format_line, 
													myre = myre,
													tag_it = self.tag_it,
													use_pos = self.use_pos
													), lines, chunksize = 2000)
				pool_instance.close()
				pool_instance.join()
				
				print(fname + ":   ", end = "")
				print(time.time() - starting)
				
				#Remove downloaded file
				os.remove("temp.txt")
			
				#Now yield from memory
				for line in lines:
									
					#Some lines have only one item#
					if len(line) > 1:
						yield line
						
			except:
				print("Problem connecting with s3, moving on")
#----------------------------------------------------------------------------------#
#Load GenSim model and build clusters using K-Means -------------------------------#

def build_clusters(model_file, nickname):

	#-------------------------------------------#
	def hash32(value):
		return hash(value) & 0xffffffff
	#-------------------------------------------#
	
	from modules.pyclustering.pyc_xmeans import xmeans
	from modules.pyclustering.pyc_center_initializer import kmeans_plusplus_initializer

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
	
	word_vectors = word_vectors[0:,1990:]
	print(word_vectors.shape)

	#Initalize a k-means object and use it to extract centroids
	print("Starting X-Means clustering.")
	initial_centers = kmeans_plusplus_initializer(word_vectors, 5).initialize()
	xmeans_instance = xmeans(word_vectors, initial_centers, kmax = 5000, ccore = False)
	xmeans_instance.process()
	clusters = xmeans_instance.get_clusters()
	
	#Get the end time and print how long the process took
	end = time.time()
	elapsed = end - start
	print("Time taken for X Means clustering: " + str(elapsed) + " seconds.")
	
	write_dict = {}
	write_pos_dict = {}
	
	for i in range(len(clusters)):
		for word in clusters[i]:
			write_dict[model.wv.index2word[word]] = i
	
	if "POS" in nickname:
	
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
				
	else:
		write_pos_dict = {}
			
	return write_dict, write_pos_dict
#-------------------------------------------------------------------------------#
#Write clusters in readable comma separated format --------------------------------#

def write_clusters(input_clusters, output_file):

	with codecs.open(output_file, "w", encoding = "utf-8") as fw:

		for key in list(set(sorted(input_clusters.values()))):
			
			print(key)
			
			for word in sorted(input_clusters.keys()):
				if input_clusters[word] == key:
					
					fw.write(word + "," + str(key) + "\n")

	return
#--------------------------------------------------------------------------------#

def write_file(output_file, model):
	
	#Save clusters#
	if os.path.isfile(output_file):
		os.remove(output_file)
		
	with open(output_file, "wb") as f:
		pickle.dump(model, f, protocol = 4)
		
	return
#--------------------------------------------------------------------------------#

def read_file(filename):
	
	with open(filename, 'rb') as f:
		model = pickle.load(f)
		
	return model
#--------------------------------------------------------------------------------#

#Prevent pool workers from starting here#
if __name__ == '__main__':
#---------------------------------------#

	nickname = "zho.LEX.2000dim.2000min.25iter"		#Name the output files
	dict_folder = "Dict_Files/zho"					#Specify the input files
	language = "zho"								#Specify language for POS tagging model
	use_pos = False									#If True, tag words first
	s3_bucket = "gsproto-lingscan"					#If not False, contains a str s3 bucket name
	
	#-----------------------------------#
	
	vector_write = os.path.join("Dict_Models", nickname + ".Vectors.p")
	model = train_model(workers = 16,
						sg = 1, 
						size = 2000, 
						min_count = 2000, 
						hs = 1, 
						iter = 20, 
						max_vocab = None, 
						language = language, 
						use_pos = use_pos,
						vector_write = vector_write,
						s3_bucket = s3_bucket,
						s3_prefix = dict_folder
						)
	
	write_dict, write_pos_dict = build_clusters(model, nickname)
	write_file(os.path.join("./", nickname + ".Clusters.p"), write_dict)
	write_file(os.path.join("./", nickname + ".Clusters.POS.p"), write_pos_dict)

	output_file = os.path.join("./", nickname + ".Clusters.txt")
	write_clusters(write_dict, output_file)
	
	output_file = os.path.join("./", nickname + ".Clusters.POS.txt")
	write_clusters(write_pos_dict, output_file)