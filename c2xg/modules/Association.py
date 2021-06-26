import time
import os
import math
import cytoolz as ct
import numpy as np
from collections import defaultdict
from functools import partial
import multiprocessing as mp
from numba import jit

from .Encoder import Encoder

#-------------------------------------------------------------------#
#The main calculation function is outside of the class for jitting

@jit(nopython = True, nogil = True)
def calculate_measures(lr_list, rl_list):

	#Mean Delta-P
	mean_lr = np.mean(lr_list)
	mean_rl = np.mean(rl_list)
	
	#Min Delta-P
	min_lr = np.amin(lr_list)
	min_rl = np.amin(rl_list)
	
	#Directional: Scalar and Categorical
	directional = np.subtract(lr_list, rl_list)
	directional_scalar = np.sum(directional)
	directional_categorical = min((directional > 0.0).sum(), (directional < 0.0).sum())
	
	#Beginning-Reduced Delta-P
	reduced_beginning_lr = (np.sum(lr_list) - np.sum(lr_list[1:]))
	reduced_beginning_rl = (np.sum(rl_list) - np.sum(rl_list[1:]))
	
	#End-Reduced Delta-P
	reduced_end_lr = (np.sum(lr_list) - np.sum(lr_list[0:-1]))
	reduced_end_rl = (np.sum(rl_list) - np.sum(rl_list[0:-1]))
	
	#Package and return
	return_list = [mean_lr, 
					mean_rl, 
					min_lr, 
					min_rl, 
					directional_scalar, 
					directional_categorical, 
					reduced_beginning_lr,
					reduced_beginning_rl,
					reduced_end_lr,
					reduced_end_rl
					]
					
	return return_list

#------------------------------------------------------------------#

class Association(object):

	def __init__(self, Loader, nickname = ""):
	
		#Initialize Ingestor
		self.language = Loader.language
		self.Encoder = Encoder(Loader = Loader)
		self.Loader = Loader
		self.nickname = nickname
		
	#--------------------------------------------------------------#
	
	def process_ngrams(self, filename, Encoder, save = False, lex_only = False):

		#Initialize bigram dictionary
		ngrams = defaultdict(int)
		unigrams = defaultdict(int)

		starting = time.time()
		total = 0

		#"filename" can be a string filename or a list of sentences
		if isinstance(filename, list):
			lines = Encoder.load_stream(filename, no_file = True)
		else:
			lines = Encoder.load_stream(filename)

		for line in lines:

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

					if lex_only == False:
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
		ngrams = {key: ngrams[key] for key in list(ngrams.keys()) if -1 not in key[0] and -1 not in key[1]}
		unigrams = {key: unigrams[key] for key in list(unigrams.keys()) if -1 not in key}
		
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
			self.Loader.save_file(ngrams, filename + "." + self.nickname + ".ngrams.p")
			return os.path.join(self.Loader.output_dir, filename + "." + self.nickname + ".ngrams.p")
				
		else:
			return ngrams
	#--------------------------------------------------------------------------------------------#

	def find_ngrams(self, files = None, workers = 1, nickname = "", save = True, lex_only = False):

		starting = time.time()
		
		if files == None:
			files = self.Loader.list_input()
		
		if save == True:
			
			pool_instance = mp.Pool(processes = workers, maxtasksperchild = 1)
			output_files = pool_instance.map(partial(self.process_ngrams, Encoder = self.Encoder, save = save), files, chunksize = 1)
			pool_instance.close()
			pool_instance.join()
			
			print("")
			print("\tFILES PROCESSED: " + str(len(output_files)))
			print("\tTOTAL TIME: " + str(time.time() - starting))
			
			return output_files

		elif save == False:
	
			ngrams = self.process_ngrams(files, Encoder = self.Encoder, save = save, lex_only = lex_only)
			
			print("")
			print("\tTOTAL TIME: " + str(time.time() - starting))
			
			return ngrams

	#---------------------------------------------------------------------------------------------#
	
	def merge_ngrams(self, files = None, ngram_dict = None, n_gram_threshold = 0):
		
		#If loading results
		if ngram_dict == None:
			all_ngrams = []
			
			#Get a list of ngram files
			if files == None and ngram_dict == None:
				files = self.Loader.list_output(type = "ngrams")
				
			#Break into lists of 20 files
			file_list = ct.partition_all(20, files)
			
			for files in file_list:
				
				ngrams = []		#Initialize holding list
				
				#Load
				for dict_file in files:
					try:
						ngrams.append(self.Loader.load_file(dict_file))
					except:
						print("Not loading " + str(dict_file))
			
				#Merge
				ngrams = ct.merge_with(sum, [x for x in ngrams])
			
				print("\tSUB-TOTAL NGRAMS: " + str(len(list(ngrams.keys()))))
				print("\tSUB-TOTAL WORDS: " + str(ngrams["TOTAL"]))
				print("\n")
				
				all_ngrams.append(ngrams)
				
			#Now merge everything
			all_ngrams = ct.merge_with(sum, [x for x in all_ngrams])


		#If loading results
		elif ngram_dict != None:
			all_ngrams = ngram_dict

		print("\tTOTAL NGRAMS: " + str(len(list(all_ngrams.keys()))))
		print("\tTOTAL WORDS: " + str(all_ngrams["TOTAL"]))
		
		#Now enforce threshold
		keepable = lambda x: x > n_gram_threshold
		all_ngrams = ct.valfilter(keepable, all_ngrams)
		
		print("\tAfter pruning:")
		print("\tTOTAL NGRAMS: " + str(len(list(all_ngrams.keys()))))
		
		return all_ngrams
	#----------------------------------------------------------------------------------------------#

	def calculate_association(self, ngrams, smoothing = False, save = False):
	
		print("\n\tCalculating association for " + str(len(list(ngrams.keys()))) + " pairs.")
		association_dict = defaultdict(dict)
		total = ngrams["TOTAL"]
		starting = time.time()

		#Loop over pairs
		for key in ngrams.keys():
		
			try:
				count = ngrams.get(key, 1)
				freq_1 = ngrams.get(key[0], 1)
				freq_2 = ngrams.get(key[1], 1)

				#a = Frequency of current pair
				a = count
				a = max(a, 0.1)
					
				#b = Frequency of X without Y
				b = freq_1 - count
				b = max(b, 0.1)
					
				#c = Frequency of Y without X
				c = freq_2 - count
				c = max(c, 0.1)
					
				#d = Frequency of units without X or Y
				d = total - a - b - c
				d = max(d, 0.1)
					
				if smoothing == False:

					association_dict[key]["LR"] = float(a / (a + c)) - float(b / (b + d))
					association_dict[key]["RL"] = float(a / (a + b)) - float(c / (c + d))
					association_dict[key]["Freq"] = count

				#Add smoothing to the Delta-P values
				elif smoothing == True:

					association_dict[key]["LR"] = float(a / (a + math.pow(c, 0.75))) - float(b / (b + d))
					association_dict[key]["RL"] = float(a / (a + math.pow(b, 0.75))) - float(c / (c + d))
					association_dict[key]["Freq"] = count
				
			except Exception as e:
				print(e, key)
				
		print("\tProcessed " + str(len(list(association_dict.keys()))) + " items in " + str(time.time() - starting))

		if save == True:
			self.Loader.save_file(association_dict, self.language  + "." + self.nickname + ".association.p")
		
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
		