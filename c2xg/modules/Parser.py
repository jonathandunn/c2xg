import time
import numpy as np
import cytoolz as ct
import multiprocessing as mp
from functools import partial
from numba import jit, int64, njit, types, typed
from scipy.sparse import coo_matrix


#--------------------------------------------------------------#
@jit(nopython = True, nogil = True)
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


def _get_candidates( unit, grammar ) : 
        # Check for: if construction[0][1] == unit[construction[0][0]-1]:
        # model_expanded[ (possible elem[0][0]) ][ elem[0][1] ].append( elem ) 
        all_plausible = list()
        X, grammar = grammar
        for elem_0_0 in X : 
                plausible = grammar[ elem_0_0 ][ unit[ elem_0_0 - 1 ] ]
                plausible = [ ( i[0], i[2] ) for i in plausible  ]
                all_plausible += plausible
        return all_plausible 
                
#--------------------------------------------------------------#

def parse_fast( line, grammar, grammar_len, sparse_matches=False ) : 
        matches = None
        if sparse_matches : 
                matches = list()
        else : 
                matches = [0 for x in range( grammar_len )]
        #Iterate over line from left to right
        for line_index in range( len( line ) ) : 
                unit = line[ line_index ] 
                #Get plausible candidates 
                candidates = _get_candidates( unit, grammar )
                for k in range(len(candidates)) : 
                        construction, grammar_index = candidates[k]
                        ## Below this is the same as the parse() function
                        match = True	#Initiate match flag to True
                        #Check each future unit in candidate
                        for j in range(1, len(construction)):
                                #If we reach the padded part of the construction, break it off
                                if construction[j] == (0,0):
                                        break
							
                                #If this unit doesn't match, stop looking
                                if line_index+j < len(line):
                                        if line[line_index+j][construction[j][0] - 1] != construction[j][1]:
                                                match = False
                                                break
						
                                #This construction is longer than the remaining line
                                else:
                                        match = False
                                        break
                        #Done with candidate
                        if match == True:
                                if sparse_matches : 
                                        matches.append( grammar_index ) 
                                else : 
                                        matches[grammar_index] += 1
        
                       
        return matches
                        
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

def _validate( lines, grammar, grammar_detailed ) : 
        from tqdm import tqdm
        for line in tqdm( lines, desc="Validating" ) : 
                matches_parse      = parse(      line, grammar=grammar )
                matches_parse_fast = parse_fast( line, grammar=grammar_detailed, grammar_len=len(grammar), sparse_matches=False )
                print( sum( matches_parse_fast ) )
                assert matches_parse == matches_parse_fast 
        return 

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

	def parse_idNet(self, lines, grammar, workers, detailed_grammar=None):
		
		#Multi-process version
		if workers != None:
			
			#First, multi-process encoded lines into memory
			pool_instance = mp.Pool(processes = workers, maxtasksperchild = None)
			lines = pool_instance.map(self.Encoder.load, lines, chunksize = 50)
			pool_instance.close()
			pool_instance.join()

			#Second, multi-process parsing
			pool_instance = mp.Pool(processes = workers, maxtasksperchild = None)
			## chunksize = int( len( list(lines) ) / workers )
			chunksize = 500

			if not detailed_grammar is None :
				lines = pool_instance.map(partial(parse_fast, grammar = detailed_grammar, grammar_len = len( grammar ), sparse_matches=False), lines, chunksize=chunksize )
			else : 
				lines = pool_instance.map(partial(parse     , grammar = grammar                                                             ), lines, chunksize=chunksize )


			pool_instance.close()
			pool_instance.join()
		
		#Single-process version
		else:

			lines = self.Encoder.load_batch(lines)
			# Uncomment to validate fast parser. 
			# _validate( lines, grammar, detailed_grammar ) 
			# print( "Validation complete", flush=True )
			# return

			if not detailed_grammar is None:
				lines = [parse_fast(line, grammar = detailed_grammar, grammar_len= len(grammar) ) for line in lines]
			else : 
				lines = [parse     (line, grammar = grammar                                   ) for line in lines]
				

				
		return lines
	#--------------------------------------------------------------#
	
	def parse_line_yield(self, lines, grammar):
		
		for line in lines:
		
			line = self.Encoder.load(line)
			line = parse(line, grammar = grammar)

			yield line
			
	#--------------------------------------------------------------#
