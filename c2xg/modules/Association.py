import time
import os
import cytoolz as ct
from collections import defaultdict
from functools import partial
import multiprocessing as mp
from modules.Encoder import Encoder

#------------------------------------------------------------------#

class Association(object):

	def __init__(self, language, Loader):
	
		#Initialize Ingestor
		self.language = language
		self.Encoder = Encoder(language = language, Loader = Loader)
		self.Loader = Loader
		
	#--------------------------------------------------------------#
	
	def process_ngrams(self, filename, Encoder, save = False):

		#Initialize bigram dictionary
		ngrams = defaultdict(int)
		unigrams = defaultdict(int)
				
		starting = time.time()
		total = 0

		for line in Encoder.load(filename):

			total += len(line)

			#Store unigrams
			for item in line:
				unigrams[(1, item[0])] += 1
				unigrams[(2, item[1])] += 1
				unigrams[(3, item[2])] += 1
			
			try:
				for bigram in ct.sliding_window(2, line):
					
					#Tuples are indexes for (LEX, POS, CAT)
					#Index types are 1 (LEX), 2 (POS), 3 (CAT)
					ngrams[((1, bigram[0][0]), (1, bigram[1][0]))] += 1	#lex_lex
					ngrams[((1, bigram[0][0]), (2, bigram[1][1]))] += 1	#lex_pos
					ngrams[((1, bigram[0][0]), (3, bigram[1][2]))] += 1	#lex_cat
					ngrams[((2, bigram[0][1]), (2, bigram[1][1]))] += 1	#pos_pos
					ngrams[((2, bigram[0][1]), (1, bigram[1][0]))] += 1	#pos_lex
					ngrams[((2, bigram[0][1]), (3, bigram[1][2]))] += 1	#pos_cat 
					ngrams[((3, bigram[0][2]), (3, bigram[1][2]))] += 1	#cat_cat
					ngrams[((3, bigram[0][2]), (2, bigram[1][1]))] += 1	#cat_pos
					ngrams[((3, bigram[0][2]), (1, bigram[1][0]))] += 1	#cat_lex
			
			#Catch errors from empty lines coming out of the encoder
			except Exception as e:
				error = e

		#Reduce nonce ngrams
		size = len(list(ngrams.keys()))
		keepable = lambda x: x > 1
		ngrams = ct.valfilter(keepable, ngrams)
		
		#Note: Keep all unigrams, they are already limited by the lexicon
		
		#Reduce null indexes
		ngrams = {key: ngrams[key] for key in list(ngrams.keys()) if 0 not in key[0] and 0 not in key[1]}
		unigrams = {key: unigrams[key] for key in list(unigrams.keys()) if 0 not in key}
		
		ngrams = ct.merge([ngrams, unigrams])	
		ngrams["TOTAL"] = total
		
		del unigrams
		
		#Print status
		print("\tTime: ", end = "")
		print(time.time() - starting, end = "")
		print(" Full: " + str(size) + " ", end = "")
		print(" Reduced: ", end = "")
		print(len(list(ngrams.keys())), end = "")
		print(" with " + str(ngrams["TOTAL"]) + " words.")
		
		if save == True:
			self.Loader.save_file(ngrams, filename + ".ngrams.p")
			return os.path.join(self.Loader.output_dir, filename + ".ngrams.p")
				
		else:
			return ngrams
	#--------------------------------------------------------------------------------------------#

	def find_ngrams(self, workers):

		print("Starting to find ngrams.")
		starting = time.time()
		
		files = self.Loader.list_input()
		
		#Multi-process#
		pool_instance = mp.Pool(processes = workers, maxtasksperchild = 1)
		output_files = pool_instance.map(partial(self.process_ngrams, Encoder = self.Encoder, save = True), files, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		
		print("")
		print("\tFILES PROCESSED: " + str(len(output_files)))
		print("\tTOTAL TIME: " + str(time.time() - starting))
		
		return output_files
	#---------------------------------------------------------------------------------------------#
	
	def merge_ngrams(self):
		
		ngrams = []		#Initialize holding list
		files = self.Loader.list_output(type = "ngrams")

		#Load
		for dict_file in files:
			ngrams.append(self.Loader.load_file(dict_file))
		
		#Merge
		ngrams = ct.merge_with(sum, [x for x in ngrams])
		
		print("\tTOTAL NGRAMS: " + str(len(list(ngrams.keys()))))
		print("\tTOTAL WORDS: " + str(ngrams["TOTAL"]))
		
		return ngrams
	#----------------------------------------------------------------------------------------------#

	def calculate_association(self, ngrams, save = False):
	
		print("Calculating association for " + str(len(list(ngrams.keys()))) + " pairs.")
		association_dict = defaultdict(dict)
		total = ngrams["TOTAL"]
		starting = time.time()

		#Loop over pairs
		for key in ngrams.keys():
			
			try:
				count = ngrams[key]
				freq_1 = ngrams[key[0]]
				freq_2 = ngrams[key[1]]
				
				#a = Frequency of current pair
				a = count
				
				#b = Frequency of X without Y
				b = freq_1 - count
				
				#c = Frequency of Y without X
				c = freq_2 - count
				
				#d = Frequency of units without X or Y
				d = total - a - b - c
				
				association_dict[key]["LR"] = float(a / (a + c)) - float(b / (b + d))
				association_dict[key]["RL"] = float(a / (a + b)) - float(c / (c + d))
				association_dict[key]["Freq"] = count
				
			except Exception as e:
				e = e
				
		print("\tProcessed " + str(len(list(association_dict.keys()))) + " items in " + str(time.time() - starting))

		if save == True:
			self.Loader.save_file(association_dict, self.language  + ".association.p")
		
		return association_dict
	#-----------------------------------------------------------------------------------------------#
	
	def get_top(self, association_dict, direction, number):
		
		#Make initial cuts without sorting to save time
		temp_dict = {key: association_dict[key][direction] for key in association_dict.keys()}
		current_threshold = 0.25
		
		while True:
		
			above_threshold = lambda x: x > current_threshold
			temp_dict = ct.valfilter(above_threshold, temp_dict)
			
			if len(list(temp_dict.keys())) > 10000:
				current_threshold = current_threshold + 0.05
				
			else:
				break
		
		#Sort and reduce
		return_list = [(key, value) for key, value in sorted(temp_dict.items(), key=lambda x: x[1], reverse = True)]
		return_list = return_list[0:number+1]

		for key, value in return_list:
			yield key, value
		