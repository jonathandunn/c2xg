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
import operator
import difflib
import copy

from .Encoder import Encoder
from .Association import Association
from .Association import calculate_measures

#--------------------------------------------------------------#

class BeamSearch(object):

	def __init__(self, delta_threshold, association_dict):
		
		#Initialize empty candidate stack
		self.candidate_stack = defaultdict(list)
		self.candidates = []
		self.search_monitor = deque(maxlen = 100)
		self.association_dict = association_dict
		self.delta_threshold = delta_threshold
		
		return
	#--------------------------------------------------------------#
	
	def beam_search(self, line):

		self.candidates = []
		
		#Loop left-to-right across the line
		for i in range(len(line)):

			#Start path from each of the current slot-constraints
			for current_start in [(1, line[i][0]), (2, line[i][1]), (2, line[i][2])]:

				#Recursive search from each available path
				self.recursive_beam(current_start, line, i, len(line))
				
		#Evaluate candidate stack
		for index in self.candidate_stack.keys():
			top_score = 0.0
			for candidate in self.candidate_stack[index]:
				current_score = self.get_score(candidate)
				if current_score > top_score:
					top_score = current_score
					top_candidate = candidate
					
			self.candidates.append(top_candidate)
		
		#Horizontal pruning
		to_pop = []
		for i in range(len(self.candidates)):
			for j in range(len(self.candidates)):
				if i != j and j > i:
					candidate1 = self.candidates[i]
					candidate2 = self.candidates[j]
					
					s = difflib.SequenceMatcher(None, candidate1, candidate2)
					largest = max([x[2] for x in s.get_matching_blocks()])
					
					if largest > 2:
						shortest = min(len(candidate1), len(candidate2))
						
						if float(largest / shortest) < 0.75:
							score1 = self.get_score(candidate1)
							score2 = self.get_score(candidate2)
							
							if score1 < score2:
								if candidate1 not in to_pop:
									to_pop.append(candidate1)
							elif candidate2 not in to_pop:
								to_pop.append(candidate2)
		
		self.candidates = [x for x in self.candidates if x not in to_pop]
		
		#Reset state
		self.candidate_stack = defaultdict(list)
		self.search_monitor = deque(maxlen = 100)

		return self.candidates
	#--------------------------------------------------------------#
	
	def recursive_beam(self, previous_start, line, i, line_length):

		go = False
		
		if len(previous_start) < 2:
			go = True
			
		if self.search_monitor.count(previous_start[0:2]) < 80:
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
									
									#Add to candidate_stack
									self.candidate_stack[i - len(current_path) + 1].append(current_path)

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
		
		return total_score
	#--------------------------------------------------------------#

class Candidates(object):

	def __init__(self, language, Loader, workers = 1, association_dict = ""):
	
		#Initialize Ingestor
		self.language = language
		self.Encoder = Encoder(Loader = Loader)
		self.Loader = Loader
		self.workers = workers
		
		if association_dict != "":
			self.association_dict = association_dict
	
	#------------------------------------------------------------------
	
	def process_file(self, filename, delta_threshold = 0.05, freq_threshold = 1, save = True):
		
		candidates = []
		starting = time.time()
		
		#Initialize Beam Search class
		BS = BeamSearch(delta_threshold, self.association_dict)
		
		for line in self.Encoder.load_stream(filename):

			if len(line) > 2:
				
				#Beam Search extraction
				candidates += BS.beam_search(line)
			
		#Count each candidate, get dictionary with candidate frequencies
		candidates = ct.frequencies(candidates)
		print("\t" + str(len(candidates)) + " candidates before pruning.")
		
		#Reduce nonce candidates
		above_zero = lambda x: x > freq_threshold
		candidates = ct.valfilter(above_zero, candidates)		
			
		#Print time and number of remaining candidates
		print("\t" + str(len(candidates)) + " candidates in " + str(time.time() - starting) + " seconds.")
	
		if save == True:
			self.Loader.save_file(candidates, filename + ".candidates.p")
			return os.path.join(self.Loader.output_dir, filename + ".candidates.p")
				
		else:
			return candidates

	#--------------------------------------------------------------#
	
	def merge_candidates(self, output_files, threshold):
		
		candidates = []
		print("Merging " + str(len(output_files)) + " files.")
		
		#Load
		for dict_file in output_files:
			try:
				candidates.append(self.Loader.load_file(dict_file))
			except Exception as e:
				print("ERROR")
				print(e)
		
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