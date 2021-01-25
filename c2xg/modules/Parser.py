import time
import numpy as np
import cytoolz as ct
import multiprocessing as mp
from functools import partial
from numba import jit, int64
from scipy.sparse import coo_matrix


#--------------------------------------------------------------#
#@jit(nopython = True, nogil = True)
def parse_examples(construction, line):

	indexes = [-1]
	matches = 0
	
	#Iterate over line from left to right
	for i in range(len(line)):
		
		unit = line[i]

		#Check if the first unit matches, to merit further consideration
		if construction[0][1] == unit[construction[0][0]-1]:
						
			match = True	#Initiate match flag to True

			#Check each future unit in candidate
			for j in range(1, len(construction)):
							
				#If we reach the padded part of the construction, break it off
				if construction[j] == (0,0):
					break
							
				#If this unit doesn't match, stop looking
				if i+j < len(line):
					if line[i+j][construction[j][0] - 1] != construction[j][1]:
										
						match = False
						break
						
				#This construction is longer than the remaining line
				else:
					match = False
					break

			#Done with candidate
			if match == True:
				matches += 1
				indexes.append(i)	#Save indexes covered by construction match
				
	return construction, indexes[1:], matches

#--------------------------------------------------------------#

@jit(nopython = True, nogil = True)
def parse_mdl_support(construction, line):

	indexes = [-1]
	matches = 0
	
	#Iterate over line from left to right
	for i in range(len(line)):
		
		unit = line[i]

		#Check if the first unit matches, to merit further consideration
		if construction[0][1] == unit[construction[0][0]-1]:
						
			match = True	#Initiate match flag to True

			#Check each future unit in candidate
			for j in range(1, len(construction)):
							
				#If we reach the padded part of the construction, break it off
				if construction[j] == (0,0):
					break
							
				#If this unit doesn't match, stop looking
				if i+j < len(line):
					if line[i+j][construction[j][0] - 1] != construction[j][1]:
										
						match = False
						break
						
				#This construction is longer than the remaining line
				else:
					match = False
					break

			#Done with candidate
			if match == True:
				matches += 1
				indexes += list(range(i, i + len(construction)))	#Save indexes covered by construction match
				
	return construction, indexes[1:], matches

#--------------------------------------------------------------#

@jit(nopython = True, nogil = True)
def parse(line, grammar):

	matches = [0 for x in range(len(grammar))]
	
	#Iterate over line from left to right
	for i in range(len(line)):
			
		unit = line[i]

		#Check for plausible candidates moving forward
		for k in range(len(grammar)):

			construction = grammar[k]	#Get construction by index
			
			#Check if the first unit matches, to merit further consideration
			if construction[0][1] == unit[construction[0][0]-1]:
						
				match = True	#Initiate match flag to True

				#Check each future unit in candidate
				for j in range(1, len(construction)):
							
					#If we reach the padded part of the construction, break it off
					if construction[j] == (0,0):
						break
							
					#If this unit doesn't match, stop looking
					if i+j < len(line):
						if line[i+j][construction[j][0] - 1] != construction[j][1]:
										
							match = False
							break
						
					#This construction is longer than the remaining line
					else:
						match = False
						break

				#Done with candidate
				if match == True:
					matches[k] += 1
	
	return matches
#--------------------------------------------------------------#

class Parser(object):

	def __init__(self, Loader, Encoder):
	
		#Initialize Parser
		self.language = Encoder.language
		self.Encoder = Encoder
		self.Loader = Loader	

	#--------------------------------------------------------------#
	
	def format_grammar(self, grammar):
	
		maxlen = max(len(i) for i in grammar)
		grammar_equal = []

		#Create a grammar with (0,0) padded items for numba
		for construction in grammar:
			new = []
			for i in range(0, maxlen):
				try:
					new.append((np.int32(construction[i][0]), np.int32(construction[i][1])))
				except:
					new.append((np.int32(0),np.int32(0)))
			
			new = tuple(new)
			grammar_equal.append(new)

		return grammar_equal
	#--------------------------------------------------------------#
	
	def parse_prep(self, files, workers = 1):

		#First, load lines into memory
		lines = []
		for file in files:
			lines += [line for line in self.Loader.read_file(file) if len(line) > 1]

		#Second, multi-process encoded lines into memory
		pool_instance = mp.Pool(processes = workers, maxtasksperchild = None)
		lines = pool_instance.map(self.Encoder.load, lines, chunksize = 2500)
		pool_instance.close()
		pool_instance.join()
		
		#Third, join lines into large numpy array
		lines = np.vstack(lines)

		return lines
	
	#--------------------------------------------------------------#
	
	def parse_batch_mdl(self, lines, grammar, freq_threshold, workers = 1):
	
		#Chunk array for workers
		total_count = len(lines)
	
		#Multi-process by construction
		pool_instance = mp.Pool(processes = workers, maxtasksperchild = None)
		results = pool_instance.map(partial(parse_mdl_support, line = lines), grammar, chunksize = 500)
		pool_instance.close()
		pool_instance.join()
		
		#Find fixed max value for match indexes
		max_matches = max([len(indexes) for construction, indexes, matches in results])
		
		#Initialize lists
		construction_list = []
		indexes_list = []
		matches_list = []
		vector_list = []
		
		#Create fixed-length arrays
		for i in range(len(results)):
			construction, indexes, matches = results[i]
			if matches > freq_threshold:
				vector_list.append(i)
				construction_list.append(construction)
				matches_list.append(matches)
				indexes_list.append(indexes)
	
		#results contains a tuple for each construction in the grammar (indexes[list], matches[int])
		return construction_list, indexes_list, np.array(matches_list), vector_list
	
	#--------------------------------------------------------------#
	
	def parse_stream(self, files, grammar):
		
		for line in self.Encoder.load_stream(files):
			matches = parse(line, grammar)
			yield matches
				
	#--------------------------------------------------------------#
	
	def parse_batch(self, files, grammar, workers):
		
		#First, load lines into memory
		lines = []
		for file in files:
			lines += [line for line in self.Loader.read_file(file) if len(line) > 1]

		#Second, multi-process encoded lines into memory
		pool_instance = mp.Pool(processes = workers, maxtasksperchild = None)
		lines = pool_instance.map(self.Encoder.load, lines, chunksize = 2500)
		pool_instance.close()
		pool_instance.join()
		
		#Third, multi-process parsing
		pool_instance = mp.Pool(processes = workers, maxtasksperchild = None)
		lines = pool_instance.map(partial(parse, grammar = grammar), lines, chunksize = 2500)
		pool_instance.close()
		pool_instance.join()
		
		return lines
				
	#--------------------------------------------------------------#

	def parse_idNet(self, lines, grammar, workers):
		
		#Multi-process version
		if workers != None:
			
			#First, multi-process encoded lines into memory
			pool_instance = mp.Pool(processes = workers, maxtasksperchild = None)
			lines = pool_instance.map(self.Encoder.load, lines, chunksize = 50)
			pool_instance.close()
			pool_instance.join()

			#Second, multi-process parsing
			pool_instance = mp.Pool(processes = workers, maxtasksperchild = None)
			lines = pool_instance.map(partial(parse, grammar = grammar), lines, chunksize = 500)
			pool_instance.close()
			pool_instance.join()
		
		#Single-process version
		else:

			lines = self.Encoder.load_batch(lines)
			lines = [parse(line, grammar = grammar) for line in lines]
				
		return lines
	#--------------------------------------------------------------#
	
	def parse_line_yield(self, lines, grammar):
		
		for line in lines:
		
			line = self.Encoder.load(line)
			line = parse(line, grammar = grammar)

			yield line
			
	#--------------------------------------------------------------#