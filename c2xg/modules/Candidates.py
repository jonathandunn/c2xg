from modules.Encoder import Encoder
import cytoolz as ct
import time
import operator
import os
import multiprocessing as mp
from functools import partial

#-------------------------------------------------------------------#
class PairsData(object):
	
	def __init__(self):
		self.index_to_association = {}
		self.index_to_bigram = {}
		self.index_to_best = {}

	def add(self, bigram, best_key, best_value):
		index = len(self.index_to_association) + 1
		self.index_to_association[index] = best_value
		self.index_to_bigram[index] = bigram
		self.index_to_best[index] = best_key
	
	def update(self, bigram, best_key, best_value):
		self.index_to_association[d["index"]] = best_value
		self.index_to_bigram[d["index"]] = bigram
		self.index_to_best[d["index"]] = best_key
		
	def pop_max(self):
		top = max(self.index_to_association.items(), key=operator.itemgetter(1))[0]
		association = self.index_to_association.pop(top)
		bigram = self.index_to_bigram.pop(top)
		best = self.index_to_best.pop(top)
		return bigram, best, association, top
		
	def len(self):
		return len(self.index_to_association)

#------------------------------------------------------------------#
class Candidates(object):

	def __init__(self, language, Loader, association_dict = ""):
	
		#Initialize Ingestor
		self.language = language
		self.Encoder = Encoder(language = language, Loader = Loader)
		self.Loader = Loader
		
		if association_dict != "":
			self.association_dict = association_dict
			
		else:
			self.association_dict = Loader.load_file(language + ".association.p")		
	
	#--------------------------------------------------------------#
	def run(self, ngrams = (3,6), workers = 1, save = False):
	
		files = self.Loader.list_input()
		
		#Multi-process#
		pool_instance = mp.Pool(processes = workers, maxtasksperchild = 1)
		pool_instance.map(partial(self.process_file, ngrams = ngrams, save = save), files, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		
		candidates = merge_candidates(self)
		
		return candidates
	#--------------------------------------------------------------#
	
	def process_file(self, filename, ngrams, save = False):

		starting = time.time()
		candidates = []
		
		for line in self.Encoder.load(filename):
		
			if len(line) > 2:
			
				#Get one-dimensional representation using the "best" version of each word (LEX, POS, CAT)
				line = self.represent_line(line)
				
				#Get list of ngrams from line
				candidates += self.ngrams_from_line(line, ngrams)
		
		#Count each candidate, get dictionary with candidate frequencies
		candidates = ct.frequencies(candidates)
		
		#Reduce nonce candidates
		above_zero = lambda x: x > 2
		candidates = ct.valfilter(above_zero, candidates)		
		
		#Print time and number of remaining candidates
		print(str(time.time() - starting)  + " " + str(len(candidates)))
		
		if save == True:
			self.Loader.save_file(ngrams, filename + ".candidates.p")
			return os.path.join(self.Loader.output_dir, filename + ".candidates.p")
			
		else:
			return candidates			
	#--------------------------------------------------------------#
	
	def ngrams_from_line(self, line, ngrams):
		
		candidates = []	#Initialize candidate list
		
		for window_size in range(ngrams[0], ngrams[1]+1):
			candidates += [x for x in ct.sliding_window(window_size, line)]
			
		return candidates
	#---------------------------------------------------------------#
	
	def represent_line(self, original_line):
	
		Pairs = PairsData()					#Special class for accessing bigrams, best representations, indexes, and association values
		line = [0 for x in original_line]	#Place selected representations in list
		
		#First, get initial pairs
		for bigram in ct.sliding_window(2, original_line):
			
			#For each bigram, find its best representation and association value
			best_key, best_value = self.get_pair(bigram)
			Pairs.add(bigram, best_key, best_value)
			
		#Third, merge adjacent pairs until further mergers are not possible
		while True:

			#Find best candidate pair for merger and remove from further consideration
			current_bigram, current_best, current_value, current_index = Pairs.pop_max()
			
			#Add best representations to line: current_index refers to rightmost member of bigram
			#If 0 present in line, representation has not yet been added
			if line[current_index] == 0:
				line[current_index] = current_best[1]
				
			if line[current_index - 1] == 0:
				line[current_index - 1] = current_best[0]
				
			#Keep going until all units have been represented	
			if 0 not in line:
				break
				
		return line
	#---------------------------------------------------------------#
	
	def check_pos(self, bigram):
		
		#Tuples are indexes for (LEX, POS, CAT)
		left_pos = bigram[0][1]
		right_pos = bigram[1][1]
				
		#By default, neither unit is a head
		result = "FAIL"
		
		#Only Left unit is a head
		if left_pos in self.Encoder.head_indexes and right_pos not in self.Encoder.head_indexes:
			result = "LEFT"
		
		#Only Right unit is a head
		elif left_pos not in self.Encoder.head_indexes and right_pos in self.Encoder.head_indexes:
			result = "RIGHT"
		
		#Both units are heads
		elif left_pos in self.Encoder.head_indexes and right_pos in self.Encoder.head_indexes:
		
			#Make a left head
			if self.Encoder.head_weights[left_pos] > self.Encoder.head_weights[right_pos]:
				result = "LEFT"
			
			#make a right head
			else:
				result = "RIGHT"
				
		return result	
	#---------------------------------------------------------------#
	
	def get_pair(self, bigram):
		
		#Tuples are indexes for (LEX, POS, CAT)
		#Index types are 1 (LEX), 2 (POS), 3 (CAT)
		candidate_list = [
			((1, bigram[0][0]), (1, bigram[1][0])),	#lex_lex
			((1, bigram[0][0]), (2, bigram[1][1])),	#lex_pos
			((1, bigram[0][0]), (3, bigram[1][2])),	#lex_cat
			((2, bigram[0][1]), (2, bigram[1][1])),	#pos_pos
			((2, bigram[0][1]), (1, bigram[1][0])),	#pos_lex
			((2, bigram[0][1]), (3, bigram[1][2])),	#pos_cat 
			((3, bigram[0][2]), (3, bigram[1][2])),	#cat_cat
			((3, bigram[0][2]), (2, bigram[1][1])),	#cat_pos
			((3, bigram[0][2]), (1, bigram[1][0])),	#cat_lex
			]
		
		#Check each candidate to see if it has the highest association
		best_value = -2										#Lowest possible value
		best_key = ((2, bigram[0][1]), (2, bigram[1][1]))	#Use POS by default
		
		for candidate in candidate_list:
			
			try:
				if self.association_dict[candidate]["LR"] > best_value:
					best_key = candidate
					best_value = self.association_dict[candidate]["LR"]
				
				elif self.association_dict[candidate]["RL"] > best_value:
					best_key = candidate
					best_value = self.association_dict[candidate]["RL"]
			
			#Some pairs aren't in the dictionary b/c not frequent enough
			except:
				counter = 0
		
		return best_key, best_value
	#--------------------------------------------------------------------------------------------#	
	
	def merge_candidates(self, output_files):
		
		candidates = []
		output_files = self.Loader.list_output(type = "candidates")
		print("Merging " + str(len(output_files)) + " files.")
		
		#Load
		for dict_file in output_files:
			candidates.append(self.Loader.load_file(dict_file))
		
		#Merge
		candidates = ct.merge_with(sum, [x for x in candidates])
		print("\tTOTAL CANDIDATES: " + str(len(list(candidates.keys()))))
		
		return candidates
	#----------------------------------------------------------------------------------------------#