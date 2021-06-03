import math
import time
import itertools
import random
import copy
from collections import deque
import cytoolz as ct
import numpy as np
import multiprocessing as mp
from functools import partial
from numba import jit
from sklearn.mixture import GaussianMixture

#---------------------------------------------------------------------------
@jit(nopython = True)
def get_construction_encoding(construction, lex_cost, pos_cost, domain_cost):

	total = 0
	
	for unit in construction:
		
		type = unit[0]
			
		if type != 0:
			if type == 1:
				total += lex_cost
					
			elif type == 2:
				total += pos_cost
					
			elif type == 3:
				total += domain_cost
		
	return total
#---------------------------------------------------------------------------
@jit(nopython = True)
def get_subset(construction_list, cost_list, match_list, pointer_list):

	l1_cost = 0
	l2_match_cost = 0
		
	#Iterate over construction in current grammar
	for i in range(len(construction_list)):
		
		#Retrieve the pre-calculated cost of this construction in the grammar
		l1_cost += cost_list[i]
						
		#Number of occurrences multiplied by the cost of encoding a pointer to this construction
		l2_match_cost += (match_list[i] * pointer_list[i])
		
	return l1_cost, l2_match_cost

#---------------------------------------------------------------------------
@jit(nopython = True)
def generate_direct(dummy_int, search, tabu_list):

	#Initialize list for indexes to change this turn
	current_move = [-1]
	tabu_additions = [-1]
	random.seed(dummy_int)
			
	while True:
			
		if len(current_move) > 50:
			break
					
		new_index = random.randint(0, len(search))
				
		if new_index not in tabu_list:
			current_move.append(new_index)
			tabu_additions.append(new_index)
					
	#Get new candidate space
	current_search = search
				
	for i in range(len(current_search)):
		if i in current_move[1:]:
			
			#Reverse candidate's current state
			if current_search[i] == 0:
				current_search[i] = 1
							
			elif current_search[i] == 1:
				current_search[i] = 0
				
	return current_search, tabu_additions[1:]
							
#---------------------------------------------------------------------------
	
class MDL_Learner(object):

	def __init__(self, Loader, Encoder, Parser, freq_threshold, vectors, candidates):
	
		print("Initializing MDL Learner for this round (loading data).")
		
		#Initialize
		self.language = Encoder.language
		self.Encoder = Encoder
		self.Loader = Loader
		self.Parser = Parser
		self.freq_threshold = freq_threshold
		self.tabu_start = False
		
		#Get fixed units costs per representation type
		self.type_cost = -math.log2(float(1.0/3.0))
		
		number_of_words = len(list(self.Encoder.word_dict.keys()))
		self.lex_cost = -math.log2(float(1.0/number_of_words))
		
		number_of_pos = len(list(self.Encoder.pos_dict.keys()))
		self.pos_cost = -math.log2(float(1.0/number_of_pos))
		
		number_of_domains = len(list(set(self.Encoder.domain_dict.values())))
		self.domain_cost = -math.log2(float(1.0/number_of_domains))
		
		#Load candidate constructions to use as grammar
		self.vectors = vectors
	
		#Reformat candidate to be equal length for numba
		self.candidates = self.Parser.format_grammar(candidates)

	#---------------------------------------------------------------------------
	
	def get_mdl_data(self, test_files, workers = None, learn_flag = True):

		#No need to reencode the test set many times
		starting = time.time()
		lines = self.Parser.parse_prep(test_files, workers = workers)	
		print("\tLoaded and encoded " + str(len(lines)) + " words in " + str(time.time() - starting))
		
		#Get {construction: indexes, matches} dictionary
		starting = time.time()
		print("\tPrepping MDL search with " + str(len(self.candidates)) + " total candidates.")
		
		self.candidates, self.indexes, self.matches, vector_list = self.Parser.parse_batch_mdl(lines, self.candidates, freq_threshold = self.freq_threshold, workers = workers)
		print("\tParsed " + str(len(lines)) + " words with " + str(len(self.candidates)) + " constructions in " + str(time.time() - starting) + " seconds.")

		self.max_index = len(lines)
		del lines	#No longer needed
		
		#Add pre-calculated construction encoding cost
		starting = time.time()
		cost_list = []
		pointer_list = []
		
		for i in range(len(self.candidates)):
			key = self.candidates[i]
			cost_list.append(get_construction_encoding(key, self.lex_cost, self.pos_cost, self.domain_cost))
			if self.matches[i] > 0:
				pointer_list.append(-math.log2(float(self.matches[i]/self.max_index)))
			else:
				pointer_list.append(0)
			
		self.costs = np.array(cost_list)
		self.pointers = np.array(pointer_list)
		self.candidates = np.array(self.candidates)
		print("\tAdded encoding costs in " + str(time.time() - starting))
		
		#Prune association vectors to include only current candidates
		if learn_flag == True:
			starting = time.time()
			self.vectors = self.vectors[vector_list,:]
			print("\tReduced association vectors to freq. threshold candidates in " + str(time.time() - starting))		

	#---------------------------------------------------------------------------
	
	def generate_searches_em(self, search, turn_limit):
	
		if search == "Initialize":
			search = [0 for i in range(12)]			#Initialize search return
		
		if self.tabu_start == False:
			self.memory = [[0,0,0,0,0,0,0,0,0,0,0,0]]		#Remember yielded searches
			self.tabu_list = deque([], maxlen = 4)			#Remember recent changes
		
		counter = 0		#Count turns per search cycle
		
		while True:
		
			#Check for tabu cycle
			if counter >= turn_limit:
				break
				
			#Get index to change
			index = random.randint(0, 11)

			#Make sure this feature hasn't been recently changed
			if index not in self.tabu_list:
				
				#Add index to tabu list
				self.tabu_list.appendleft(index)

				#Save current state in case change is rejected
				old_search = search
				
				#Make 1s into 0s
				if search[index] == 1:
					search[index] = 0
				
				#Make 0s into 1s
				elif search[index] == 0:
					search[index] = 1
	
				if search.count(1) > 2:
					if str(search) not in self.memory:
						self.memory.append(str(search))
						counter += 1
						yield [i for i in range(len(search)) if search[i] == 1]					
					
					#Combination already yielded, reject search
					else:
						search = old_search
					
				#Not enough features, reject search
				else:
					search = old_search		
		
	#---------------------------------------------------------------------------		
	
	def search(self, turn_limit = 10, workers = 1):
	
		self.search_em(turn_limit, workers)
		self.search_direct(turn_limit*3, workers)
		
		#Now return a dictionary with construction:count pairs
		candidate_dict = {}
		for i in range(len(self.candidates)):
			candidate_dict[self.candidates[i]] = self.matches[i]
		
		return candidate_dict		
		
	#---------------------------------------------------------------------------
	
	def search_em(self, turn_limit = 10, workers = 1):
	
		print("\nStarting tabu search with GMM / EM selection.\n")

		improvement_counter = 0			#Turns since improvement counter
		total_counter = 0
		best_search = "Initialize"			#Initialize search space
		best_mdl = 999999999999999999999999999999999999999999999999.0
		return_mdl = best_mdl
		
		#Iterate over search turns
		while True:
		
			#Initialize result holders
			search_list = []	
			subset_list = []
			
			if improvement_counter > 10:
				break
		
			total_counter += 1
			print("\n\tEM Search #" + str(total_counter))
			
			#Test for current best move
			starting = time.time()
			for search_space in self.generate_searches_em(best_search, turn_limit = turn_limit):
				
				#Get search space
				count_dict, labels = self.learn_subset(search_space)
				
				#Take smallest cluster
				if count_dict[0] < count_dict[1]:
					subset = [i for i in range(len(labels)) if labels[i] == 0]
				else:
					subset = [i for i in range(len(labels)) if labels[i] == 1]
				
				#Save current subset
				subset_list.append(subset)
				search_list.append(search_space)
			
			#Multi-process MDL evaluation
			mdl_list = [self.evaluate_subset(x) for x in subset_list]
			# pool_instance = mp.Pool(processes = workers, maxtasksperchild = 1)
			# mdl_list = pool_instance.map(self.evaluate_subset, subset_list, chunksize = 1)
			# pool_instance.close()
			# pool_instance.join()		
				
			#Find lowest turn of all evaluated moves
			lowest = 99999999999999999999999999999999.0
			best_move = 0
			for i in range(len(mdl_list)):
				if mdl_list[i] < lowest:
					best_move = i
					lowest = mdl_list[i]
					
			turn_mdl = mdl_list[best_move]
			search = search_list[best_move]
			turn_subset = subset_list[best_move]
			
			print("\tTurn Best Score: " + str(turn_mdl)	+ " in " + str(time.time() - starting) + " with " + str(len(turn_subset)))
			
			if turn_mdl < return_mdl:
				best_mdl = turn_mdl
				best_search = [1 if i in search else 0 for i in range(12)]
				improvement_counter = 0
				print("\tNew best search: " + str(best_search) + " with " + str(len(turn_subset)) + " candidates.")
				return_mdl = copy.deepcopy(best_mdl)
				return_search = copy.deepcopy(best_search)
				return_subset = copy.deepcopy(turn_subset)
				
			else:
				best_mdl = turn_mdl
				best_search = [1 if i in search else 0 for i in range(12)]
				improvement_counter += 1
				print("\tFailed to find new best: " + str(improvement_counter) + " with current best " + str(return_mdl) + " with " + str(len(return_subset)))
				
		#Search is finished
		print("Final best MDL: " + str(return_mdl))
		print("Final best search: " + str(return_search))
		
		#Clear search
		del self.memory
		del self.tabu_list
		self.tabu_start = False
		
		#Reduce MDL prep data
		self.candidates = self.candidates[return_subset]
		self.costs = self.costs[return_subset]
		self.matches = self.matches[return_subset]
		self.pointers = self.pointers[return_subset]
	
	#---------------------------------------------------------------------------
	
	def evaluate_subset(self, subset, return_detail = False):
	
		#External call to looping function
		if subset == False:
			l1_cost, l2_match_cost = get_subset(self.candidates, 
													self.costs, 
													self.matches, 
													self.pointers
													)
													
		else:
			l1_cost, l2_match_cost = get_subset(self.candidates[subset], 
													self.costs[subset], 
													self.matches[subset], 
													self.pointers[subset]
													)
		
		#Find unencoded indexes
		if subset == False:
			unencoded_indexes = list(ct.concat([self.indexes[i] for i in range(len(self.indexes))]))
			unencoded_indexes = self.max_index - len(list(ct.unique(unencoded_indexes)))
		
		else:
			unencoded_indexes = list(ct.concat([self.indexes[i] for i in subset]))
			unencoded_indexes = self.max_index - len(list(ct.unique(unencoded_indexes)))

		#Use unencoded indexes to get regret cost
		#Regret cost applied twice, once for encoding and once for grammar
		if unencoded_indexes > 0:
			if subset == False:
				unencoded_cost = -math.log2(float(1.0/(unencoded_indexes)))
				l2_regret_cost = (unencoded_cost * unencoded_indexes) * 2

			else:
				unencoded_cost = -math.log2(float(1.0/(unencoded_indexes + len(subset))))
				l2_regret_cost = (unencoded_cost * unencoded_indexes) * 2
		
		else:
			l2_regret_cost = 0
		
		#Total all terms
		total_mdl = l1_cost + l2_match_cost + l2_regret_cost
				
		#DEBUGGING
		if return_detail == False:
			print("\t\tMDL: " + str(total_mdl))
			print("\t\tL1 Cost: " + str(l1_cost))
			print("\t\tL2 Match Cost: " + str(l2_match_cost))
			print("\t\tL2 Regret Cost: " + str(l2_regret_cost))
			print("\t\tEncoded: " + str(self.max_index - unencoded_indexes))
			print("\t\tUnencoded: " + str(unencoded_indexes))
		
		#Calculate baseline
		if subset == False:
			baseline_cost_per = -math.log2(float(1.0/self.max_index))
			baseline_mdl = baseline_cost_per * self.max_index

			if return_detail == False:
				print("\t\tBaseline: " + str(baseline_mdl))
				print("\t\tRatio: " + str(total_mdl/baseline_mdl))		
		
		if return_detail == False:
			return total_mdl

		else:
			return total_mdl, l1_cost, l2_match_cost, l2_regret_cost, baseline_mdl
		
	#---------------------------------------------------------------------------

	def learn_subset(self, search_space):
	
		#Mask undesired features
		current_array = self.vectors[:,search_space]
	
		GM = GaussianMixture(n_components = 2, 
							covariance_type = "full", 
							tol = 0.001, 
							reg_covar = 1e-06, 
							max_iter = 1000, 
							n_init = 25, 
							init_params = "kmeans", 
							weights_init = None, 
							means_init = None, 
							precisions_init = None, 
							random_state = None, 
							warm_start = False, 
							verbose = 0, 
							verbose_interval = 10
							)
							
		GM.fit(current_array)

		labels = GM.predict(current_array)
		unique, counts = np.unique(labels, return_counts = True)
		count_dict = dict(zip(unique, counts))
		
		return count_dict, labels
	
	#---------------------------------------------------------------------------
	
	def generate_searches_direct(self, search, turn_limit, workers):
	
		if self.tabu_start == False:
			print("\t\tInitializing tabu list.")
			self.tabu_list = deque([1], maxlen = 50 * turn_limit * 2)	#Remember recent changes
			self.tabu_start = True
		
		#Multi-process turn generation
		send_list = [i for i in range(turn_limit)]
		pool_instance = mp.Pool(processes = workers, maxtasksperchild = None)
		results = pool_instance.map(partial(generate_direct, 
												search = search,
												tabu_list = list(self.tabu_list)
												), send_list, chunksize = 4)
		pool_instance.close()
		
		#Add to tabu_list and then return
		for current_search, tabu_additions in results:

			[self.tabu_list.appendleft(i) for i in tabu_additions]

			yield current_search

	#---------------------------------------------------------------------------
	
	def search_direct(self, turn_limit, workers):

		print("\nStarting direct tabu search over remaining " + str(len(self.candidates)) + " candidates.\n")
	
		improvement_counter = 0										#Turns since improvement counter
		total_counter = 0
		best_search = [1 for i in range(len(self.candidates))]		#Search for including all candidates
		best_subset = [i for i in range(len(self.candidates))]		#Subset mask for all current candidates
		best_mdl = self.evaluate_subset(best_subset)				#Baseline MDL
		baseline_mdl = best_mdl
		return_mdl = best_mdl
		
		#Iterate over search turns
		while True:
		
			#Initialize result holders
			search_list = []	
			subset_list = []
			
			if improvement_counter > 10:
				break
		
			total_counter += 1
			print("\n\tDirect Search #" + str(total_counter))
			
			#Test for current best move
			starting = time.time()
			for new_search in self.generate_searches_direct(best_search, turn_limit = turn_limit, workers = workers):
				
				#Get new subset from new candidate space
				new_subset = [i for i in range(len(new_search)) if new_search[i] == 1]

				#Save current subset
				subset_list.append(new_subset)
				search_list.append(new_search)

			#Multi-process MDL evaluation
			direct_workers = int(workers / 2)
			if direct_workers > len(subset_list):
				direct_workers = len(subset_list)
				
			print("\t\tGenerated moves in " + str(time.time() - starting) + ". Starting MDL evaluation with n workers = " + str(direct_workers))
			# pool_instance = mp.Pool(processes = direct_workers, maxtasksperchild = 1)
			# mdl_list = pool_instance.map(self.evaluate_subset, subset_list, chunksize = 1)
			# pool_instance.close()
			# pool_instance.join()
			mdl_list = [self.evaluate_subset(x) for x in subset_list]
			print("\t\tDone multi-processing; MDL_Learner line 486")
			
			#Find lowest turn of all evaluated moves
			lowest = 99999999999999999999999999999999.0
			best_move = 0
			for i in range(len(mdl_list)):
				if mdl_list[i] < lowest:
					best_move = i
					lowest = mdl_list[i]
					
			turn_mdl = mdl_list[best_move]
			turn_search = search_list[best_move]
			turn_subset = subset_list[best_move]
			
			print("\tTurn Best Score: " + str(turn_mdl)	+ " in " + str(time.time() - starting) + " against baseline "  + str(baseline_mdl))	
			
			if turn_mdl < return_mdl:
				best_mdl = turn_mdl
				best_search = turn_search
				improvement_counter = 0
				
				print("\tNew best search: " + str(len(best_search)) + " total with " + str(len(turn_subset)) + " current candidates.")
				return_mdl = best_mdl
				return_search = best_search
				return_subset = turn_subset
				
			else:
				best_mdl = turn_mdl
				best_search = turn_search
				improvement_counter += 1
				print("\tFailed to find new best: " + str(improvement_counter))
				
		#Search is finished
		print("Final best MDL: " + str(return_mdl))
		print("Final best search: " + str(len(return_search)))
		print("Starting baseline MDL: " + str(baseline_mdl))
		
		#Clear search
		del self.tabu_list
		
		#Reduce MDL prep data
		self.candidates = self.candidates[return_subset]
		self.matches = self.matches[return_subset]
		del self.costs
		del self.pointers