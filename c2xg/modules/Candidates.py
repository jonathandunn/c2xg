import time
import operator
import os
import numpy as np
import pandas as pd
import cytoolz as ct
import multiprocessing as mp
from functools import partial
from numba import jit
from collections import defaultdict
from collections import deque
import itertools
import operator
import difflib

try:
	from modules.Encoder import Encoder
	from modules.Association import Association
	from modules.Association import calculate_measures
except:
	from c2xg.modules.Encoder import Encoder
	from c2xg.modules.Association import Association
	from c2xg.modules.Association import calculate_measures

#--------------------------------------------------------------#
def find_candidates(Loader, Encoder, language, workers = 1, files = "", save = True, association_dict = None):

	if files == "":
		files = Loader.list_input()
		
	print("Initializing module")
	C = Candidates(language, Loader, association_dict)

	print("Starting multi-process")
	#Multi-process#
	pool_instance = mp.Pool(processes = workers, maxtasksperchild = 1)		
	pool_instance.map(partial(C.process_file), files, chunksize = 1)
	pool_instance.close()
	pool_instance.join()
		
	return
#--------------------------------------------------------------#	
#--------------------------------------------------------------#
	
def represent_line(original_line):
	
	Pairs = PairsData()					#Special class for accessing bigrams, best representations, indexes, and association values
	line = [0 for x in original_line]	#Place selected representations in list
		
	#First, get initial pairs
	for bigram in ct.sliding_window(2, original_line):
			
		#For each bigram, find its best representation and association value
		best_key, best_value = get_pair(bigram)
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
	
def get_pair(bigram):
		
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
			if association_dict[candidate]["LR"] > best_value:
				best_key = candidate
				best_value = association_dict[candidate]["LR"]
				
			elif association_dict[candidate]["RL"] > best_value:
				best_key = candidate
				best_value = association_dict[candidate]["RL"]
			
		#Some pairs aren't in the dictionary b/c not frequent enough
		except:
			counter = 0
		
	return best_key, best_value
#--------------------------------------------------------------------------------------------#

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
		self.Encoder = Encoder(Loader = Loader)
		self.Loader = Loader
		self.delta_threshold = 0.05
		self.candidate_scores = {}
		self.search_monitor = deque(maxlen = 20)
		
		if association_dict != "":
			self.association_dict = association_dict
	
	#------------------------------------------------------------------
	def process_file(self, filename):
		
		candidates = []
		starting = time.time()
		
		for line in self.Encoder.load_stream(filename):

			if len(line) > 2:
				
				#Get one-dimensional representation using the "best" version of each word (LEX, POS, CAT)
				#line = represent_line(line)
				#candidates += ngrams_from_line(line, ngrams)
				
				#Beam Search extraction
				candidates += self.beam_search(line)
				
		#Count each candidate, get dictionary with candidate frequencies
		candidates = ct.frequencies(candidates)
			
		#Reduce nonce candidates
		above_zero = lambda x: x > 2
		candidates = ct.valfilter(above_zero, candidates)		
			
		#Print time and number of remaining candidates
		print("\t" + str(len(candidates)) + " candidates in " + str(time.time() - starting) + " seconds.")
			
		if save == True:
			self.Loader.save_file(candidates, filename + ".candidates.p")
			return os.path.join(self.Loader.output_dir, filename + ".candidates.p")
				
		else:
			return candidates
	#--------------------------------------------------------------#
	
	def beam_search(self, line):

		#Initialize empty candidate stack
		self.candidate_stack = defaultdict(dict)
		candidates = []
		
		#Loop left-to-right across the line
		for i in range(len(line)):

			#Start path from each of the current slot-constraints
			for current_start in [(1, line[i][0]), (2, line[i][1]), (2, line[i][2])]:

				#Recursive search from each available path
				self.recursive_beam(current_start, line, i, len(line))
				
		#Evaluate candidate stack
		for index in self.candidate_stack.keys():
			top_score = 0.0
			for candidate in self.candidate_stack[index].keys():
				if self.candidate_stack[index][candidate] > top_score:
					top_score = self.candidate_stack[index][candidate]
					top_candidate = candidate
					
			candidates.append(top_candidate)
		
		#Horizontal pruning
		to_pop = []
		for i in range(len(candidates)):
			for j in range(len(candidates)):
				if i != j and j > i:
					candidate1 = candidates[i]
					candidate2 = candidates[j]
					
					s = difflib.SequenceMatcher(None, candidate1, candidate2)
					largest = max([x[2] for x in s.get_matching_blocks()])
					
					if largest > 2:
						shortest = min(len(candidate1), len(candidate2))
						
						if float(largest / shortest) < 0.75:
							score1 = self.candidate_scores[candidate1]
							score2 = self.candidate_scores[candidate2]
							
							if score1 < score2:
								if candidate1 not in to_pop:
									to_pop.append(candidate1)
							elif candidate2 not in to_pop:
								to_pop.append(candidate2)
		
		candidates = [x for x in candidates if x not in to_pop]
	
		return candidates
	#--------------------------------------------------------------#
	
	def recursive_beam(self, previous_start, line, i, line_length):

		go = False
		
		if len(previous_start) < 2:
			go = True
			
		if self.search_monitor.count(previous_start[0:2]) < 19:
			go = True
			
			
		if go == True:
			self.search_monitor.append(previous_start[0:2])
			#Progress down the line
			i += 1

			#Stop at the end
			if i < line_length:
				
				#For each available next path
				for start in [(1, line[i][0]), (2, line[i][1]), (3, line[i][2])]:
					
					#Create larger path
					try:
						previous_start = list(ct.concat(previous_start))

					except:
						previous_start = previous_start
						
					current_path = list(ct.concat([previous_start, start]))
					current_path = tuple(ct.partition(2, current_path))
					
					if len(current_path) > 2:
						test_path = current_path[-2:]
						current_dict = self.association_dict[test_path]
							
						if current_dict != {}:
									
							delta_p = max(current_dict["LR"], current_dict["RL"])
								
							if delta_p > self.delta_threshold:
								self.recursive_beam(current_path, line, i, line_length)
															
							#This is the end of a candidate sequence
							else:
								#Has to be at least 3 slots
								if len(current_path) > 3:
										
									#Remove the bad part
									current_path = current_path[0:-1]
									
									#Get score if not cached
									if current_path not in self.candidate_scores:
										self.get_score(current_path)

									#Add to candidate_stack
									self.candidate_stack[i - len(current_path) + 1][current_path] = self.candidate_scores[current_path]

					else:
						current_dict = self.association_dict[current_path]

						if current_dict != {}:
							delta_p = max(current_dict["LR"], current_dict["RL"])
								
							if delta_p > self.delta_threshold:
								self.recursive_beam(current_path, line, i, line_length)
								
			return
	#--------------------------------------------------------------#
	
	def get_score(self, current_candidate):
	
		total_score = 0.0
		
		for pair in ct.sliding_window(2, current_candidate):
		
			current_dict = self.association_dict[pair]
			current_score = max(current_dict["RL"], current_dict["LR"])
			total_score += current_score
	
		self.candidate_scores[current_candidate] = total_score
		
		return
	#--------------------------------------------------------------#
	
	def ngrams_from_line(line, ngrams):
			
		candidates = []	#Initialize candidate list
			
		for window_size in range(ngrams[0], ngrams[1]+1):
			candidates += [x for x in ct.sliding_window(window_size, line)]
				
		return candidates
	#--------------------------------------------------------------#
	
	def merge_candidates(self, output_files, threshold):
		
		candidates = []
		print("Merging " + str(len(output_files)) + " files.")
		
		#Load
		for dict_file in output_files:
			candidates.append(self.Loader.load_file(dict_file))
		
		#Merge
		candidates = ct.merge_with(sum, [x for x in candidates])
		print("\tTOTAL CANDIDATES BEFORE PRUNING: " + str(len(list(candidates.keys()))))
		
		#Prune
		above_threshold = lambda x: x > threshold
		candidates = ct.valfilter(above_threshold, candidates)
		print("\tTOTAL CANDIDATES AFTER PRUNING: " + str(len(list(candidates.keys()))))
		
		return candidates
	#----------------------------------------------------------------------------------------------#
	
	def get_association(self, candidate_dict, association_dict = False, save = False):
	
		#Initialize Association module
		Assoc = Association(Loader = self.Loader)
		
		if association_dict == False:
			try:
				self.association_dict = self.Loader.load_file(self.language + ".association.p")
				
			except Exception as e:
				print(e)
				print("Missing pairwise counts. Need to run Association.calculate_association() first")
				sys.kill()
				
		else:
			self.association_dict = association_dict
		
		#Process candidatess
		starting = time.time()
		candidates = list(candidate_dict.keys())
		results = [self.get_pairwise_lists(candidate) for candidate in candidates]
		print(str(len(list(candidate_dict.keys()))) + " candidates in " + str(time.time() - starting) + " seconds.")
			
		#Define the DataFrame columns
		columns = ["candidate", "mean_lr", "mean_rl", "min_lr", "min_rl", "directional_scalar",
					"directional_categorical", "reduced_beginning_lr", "reduced_beginning_rl",
					"reduced_end_lr", "reduced_end_rl", "endpoint_lr", "endpoint_rl"]
		
		results = np.array(results)
		print(results.shape)
		
		if save == True:
			self.Loader.save_file((candidates, results), self.language + ".candidates_association.p")
			
		return results
	#----------------------------------------------------------------------------------------------#
		
	def get_pairwise_lists(self, candidate):

		lr_list = []	#Initiate list of LR association values
		rl_list = []	#Initiate list of RL association values
		
		#Populate the pairwise value lists
		for current_pair in ct.sliding_window(2, candidate):

			lr_list.append(self.association_dict[current_pair]["LR"])
			rl_list.append(self.association_dict[current_pair]["RL"])

		#Send lists to class-external jitted function for processing
		return_list = calculate_measures(np.array(lr_list), np.array(rl_list))
		
		#Check for end-point
		try:
			endpoint_lr = self.association_dict[(candidate[0], candidate[-1])]["LR"]
			endpoint_rl = self.association_dict[(candidate[0], candidate[-1])]["RL"]
			
		except Exception as e:
			endpoint_lr = 0.0
			endpoint_rl = 0.0
			
		#Add Endpoint to return_list
		return_list.append(endpoint_lr)
		return_list.append(endpoint_rl)
		
		#return_list contains the following items:
		#--- candidate (representation, index) tuples
		#--- mean_lr 
		#--- mean_rl
		#--- min_lr
		#--- min_rl
		#--- directional_scalar
		#--- directional_categorical
		#--- reduced_beginning_lr
		#--- reduced_beginning_rl
		#--- reduced_end_lr
		#--- reduced_end_rl
		#--- endpoint_lr
		#--- endpoint_rl
		
		return return_list
	#----------------------------------------------------------------------------------------------#