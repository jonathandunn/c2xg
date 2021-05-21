import os
import random
import pickle
import copy
import operator
import codecs
from collections import defaultdict
import multiprocessing as mp
from functools import partial
from pathlib import Path
from cleantext import clean


try : 
        from .modules.Encoder import Encoder
        from .modules.Loader import Loader
        from .modules.Parser import Parser
        from .modules.Association import Association
        from .modules.Candidates import Candidates
        from .modules.MDL_Learner import MDL_Learner
        from .modules.Parser import parse_examples

except : 
        from modules.Encoder import Encoder
        from modules.Loader import Loader
        from modules.Parser import Parser
        from modules.Association import Association
        from modules.Candidates import Candidates
        from modules.MDL_Learner import MDL_Learner
        from modules.Parser import parse_examples


#------------------------------------------------------------

def eval_mdl(files, workers, candidates, Load, Encode, Parse, freq_threshold = -1, report = False):
	
	print("Now initiating MDL evaluation: " + str(files))

	for file in files:
		print("\tStarting " + str(file))		
		MDL = MDL_Learner(Load, Encode, Parse, freq_threshold = freq_threshold, vectors = {"na": 0}, candidates = candidates)
		MDL.get_mdl_data([file], workers = workers, learn_flag = False)
		current_mdl = MDL.evaluate_subset(subset = False)
			
	if report == True:
		return current_mdl
#------------------------------------------------------------		

def delta_grid_search(candidate_file, test_file, workers, mdl_workers, association_dict, freq_threshold, language, in_dir, out_dir, s3, s3_bucket, max_words, nickname = "current"):
	
	print("\nStarting grid search for beam search settings.")
	result_dict = {}
		
	delta_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
	
	if len(delta_thresholds) < workers:
		parse_workers = len(delta_thresholds)
	else:
		parse_workers = workers
		
	#Multi-process#
	pool_instance = mp.Pool(processes = parse_workers, maxtasksperchild = 1)
	distribute_list = [(x, candidate_file) for x in delta_thresholds]

	pool_instance.map(partial(process_candidates, 
								association_dict = association_dict.copy(),
								language = language,
								freq_threshold = freq_threshold,
								in_dir = in_dir,
								out_dir = out_dir,
								s3 = s3, 
								s3_bucket = s3_bucket,
								max_words = max_words,
								nickname = nickname
								), distribute_list, chunksize = 1)
	pool_instance.close()
	pool_instance.join()
				
	#Now MDL
	if language == "zho":
		zho_split = True
	else:
		zho_split = False
		
	Load = Loader(in_dir, out_dir, language, s3, s3_bucket, max_words = max_words)
	Encode = Encoder(Loader = Load, zho_split = zho_split)
	Parse = Parser(Load, Encode)
	
	for threshold in delta_thresholds:
		print("\tStarting MDL search for " + str(threshold))
		filename = str(candidate_file) + "." + nickname + ".delta." + str(threshold) + ".p"
		candidates = Load.load_file(filename)
		
		if len(candidates) < 5:
			print("\tNot enough candidates!")
		
		else:

			mdl_score = eval_mdl(files = test_file, 
									candidates = candidates, 
									workers = mdl_workers, 
									Load = Load, 
									Encode = Encode, 
									Parse = Parse, 
									freq_threshold = freq_threshold, 
									report = True
									)
			
			result_dict[threshold] = mdl_score
			print("\tThreshold: " + str(threshold) + " and MDL: " + str(mdl_score))
		
	#Get threshold with best score
	print(result_dict)
	best = min(result_dict.items(), key=operator.itemgetter(1))[0]

	#Get best candidates
	filename = str(candidate_file) + "." + nickname + ".delta." + str(best) + ".p"
	best_candidates = Load.load_file(filename)
		
	return best, best_candidates

#------------------------------------------------------------

def process_candidates(input_tuple, association_dict, language, in_dir, out_dir, s3, s3_bucket, freq_threshold = 1, mode = "", max_words = False, nickname = "current"):

	threshold =  input_tuple[0]
	candidate_file = input_tuple[1]
	
	print("\tStarting " + str(threshold) + " with freq threshold: " + str(freq_threshold))
	Load = Loader(in_dir, out_dir, language, s3, s3_bucket, max_words)
	C = Candidates(language = language, Loader = Load, association_dict = association_dict)
	
	if mode == "candidates":
		filename = str(candidate_file + ".candidates.p")
		
	else:
		filename = str(candidate_file) + "." + nickname + ".delta." + str(threshold) + ".p"
	
	if filename not in Load.list_output():
	
		candidates = C.process_file(candidate_file, threshold, freq_threshold, save = False)
		Load.save_file(candidates, filename)
	
	#Clean
	del association_dict
	del C
	
	return

#-------------------------------------------------------------------------------

class C2xG(object):
	
	def __init__(self, data_dir, language, s3 = False, s3_bucket = "", nickname = "", model = "", zho_split = False, max_words = False, fast_parse=False):
	
		#Initialize
		print("Initializing C2xG")
		in_dir = os.path.join(data_dir, "IN")
		self.nickname = nickname
		print("Current nickname: " + nickname)
		out_dir = os.path.join(data_dir, "OUT")
			
		self.language = language
		self.zho_split = zho_split
		self.Load = Loader(in_dir, out_dir, language = self.language, s3 = s3, s3_bucket = s3_bucket, max_words = max_words)
		self.Encode = Encoder(Loader = self.Load, zho_split = self.zho_split)
		self.Association = Association(Loader = self.Load, nickname = self.nickname)
		self.Candidates = Candidates(language = self.language, Loader = self.Load)
		self.Parse = Parser(self.Load, self.Encode)
		
		self.in_dir = in_dir
		self.out_dir = out_dir
		self.s3 = s3
		self.s3_bucket = s3_bucket
		self.max_words = max_words

		#Try to load default or specified model
		if model == "":
			model = self.language + ".Grammar.v1.p"
		
		try:
			modelname = Path(__file__).parent / os.path.join("data", "models", model)
			with open(modelname, "rb") as handle:
				self.model = pickle.load(handle)
		
		except:
			print("No model exists, loading empty model.")
			self.model = None
			
		if fast_parse : 
			self._detail_model() ## self.detailed_model set by this. 
		else : 
			self.detailed_model = None

		#self.n_features = len(self.model)
		self.Encode.build_decoder()
		
	#------------------------------------------------------------------
		

	def _detail_model( self ) : 
                ## Update model so we can access grammar faster ... 
                ## Want to make `if construction[0][1] == unit[construction[0][0]-1]` faster
                ## Dict on construction[0][1] which is self.model[i][0][1] (Call this Y)
                ## BUT unit[ construction[0][0] - 1 ] changes with unit ... 
                ## construction[0][0] values are very limited.  (call this X)
                ## dict[ construction[0][0] ][ construction[0][1] ] = list of constructions
                model_expanded = dict()
                X = list( set( [ self.model[i][0][0] for i in range(len(self.model)) ] ) )
                for x in X : 
                        model_expanded[ x ] = defaultdict( list ) 
                        this_x_elems = list()
                        for k, elem in enumerate( self.model ) : 
                                if elem[0][0] != x : 
                                        continue
                                elem_trunc = [ i for i in elem if i != (0,0) ]
                                model_expanded[ x ][ elem[0][1] ].append( ( elem, elem_trunc, k ) )
                self.detailed_model = ( X, model_expanded ) 


	#------------------------------------------------------------------
		
	def parse_return(self, input, mode = "files", workers = 1):

		#Compatbility with idNet
		if mode == "idNet":
			mode = "lines"
			
		#Make sure grammar is loaded
		if self.model == None:
			print("Unable to parse: No grammar model provided.")
			sys.kill()
			
		#Accepts str of filename or list of strs of filenames
		if isinstance(input, str):
			input = [input]
		
		#Text as input
		if mode == "lines":
			lines = self.Parse.parse_idNet(input, self.model, workers, self.detailed_model )
			return lines	
					
		#Filenames as input
		elif mode == "files":
		
			features = self.Parse.parse_batch(input, self.model, workers)
			return features
		
	#-------------------------------------------------------------------------------
	
	def parse_yield(self, input, mode = "files"):

		#Make sure grammar is loaded
		if self.model == None:
			print("Unable to parse: No grammar model provided.")
			sys.kill()
			
		#Accepts str of filename or list of strs in batch/stream modes
		if isinstance(input, str):
			input = [input]
		
		#Filenames as input
		if mode == "files":
			for features in self.Parse.parse_stream(input, self.model):
				yield features

		#Texts as input
		elif mode == "lines":
		
			for line in input:
				line = self.Parse.parse_line_yield(line, self.model)
				yield line			
			
	#-------------------------------------------------------------------------------
	def print_constructions(self):

		for i in range(len(self.model)):
			
			x = self.model[i]
			printed_examples = []

			#Prune to actual constraints
			x = [y for y in x if y[0] != 0]
			length = len(x)
			construction = self.Encode.decode_construction(x)

			print(i, construction)

		return
	#-------------------------------------------------------------------------------
	def print_examples(self, input_file, output_file, n):

		#Read and write in the default data directories
		output_file = os.path.join(self.out_dir, output_file)

		#Save the pre-processed lines, to save time later
		line_list = []
		for line, encoding in self.Encode.load_examples(input_file):
			line_list.append([line, encoding])

		with codecs.open(output_file, "w", encoding = "utf-8") as fw:
			for i in range(len(self.model)):
			
				x = self.model[i]
				printed_examples = []

				#Prune to actual constraints
				x = [y for y in x if y[0] != 0]
				length = len(x)
				construction = self.Encode.decode_construction(x)

				print(i, construction)
				fw.write(str(i) + "\t")
				fw.write(construction)
				fw.write("\n")

				#Track how many examples have been found
				counter = 0

				for line, encoding in line_list:

					construction_thing, indexes, matches = parse_examples(x, encoding)

					if matches > 0:
						for index in indexes:
							
							text = line.split()[index:index+length]

							if text not in printed_examples:
								counter += 1
								printed_examples.append(text)
								fw.write("\t" + str(counter) + "\t" + str(text) + "\n")
					
					#Stop looking for examples at threshold
					if counter > n:
						break
				
				#End of examples for this construction
				fw.write("\n\n")
	#-------------------------------------------------------------------------------
	
	def get_lexicon(self, file):

		vocab = []

		for line in self.Load.read_file(file):
			
			#Use clean-text
			line = clean(line,
							fix_unicode = True,
							to_ascii = False,
							lower = True,
							no_line_breaks = True,
							no_urls = True,
							no_emails = True,
							no_phone_numbers = True,
							no_numbers = True,
							no_digits = True,
							no_currency_symbols = True,
							no_punct = True,
							replace_with_punct = "",
							replace_with_url = "<URL>",
							replace_with_email = "<EMAIL>",						
							replace_with_phone_number = "<PHONE>",
							replace_with_number = "<NUMBER>",
							replace_with_digit = "0",
							replace_with_currency_symbol = "<CUR>"
							)

			line = line.split()
			vocab += line

		return set(vocab)

	#-------------------------------------------------------------------------------		
	
	def learn(self, 
				nickname, 
				cycles = 1, 
				cycle_size = (1, 5, 20), 
				freq_threshold = 10, 
				beam_freq_threshold = 10,
				turn_limit = 10, 
				workers = 1,
				mdl_workers = 1,
				states = None,
				fixed_set = [],
				beam_threshold = None,
				no_mdl = False,
				):
	
		self.nickname = nickname

		#Check learning state and resume
		self.model_state_file = self.language + "." + self.nickname + ".State.p"
		
		try:
			loader_files = self.Load.list_output()
		except:
			loader_files = []

		if self.model_state_file in loader_files:
			print("Resuming learning state.")
			self.progress_dict, self.data_dict = self.Load.load_file(self.model_state_file)
			
			if states != None:
				print("Manual state change!")
				for state in states:
					self.progress_dict[state[0]][state[1]] = state[2]
			
		else:
			print("Initializing learning state.")
			self.data_dict = self.divide_data(cycles, cycle_size, fixed_set)
			self.progress_dict = self.set_progress()
			self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
		
		#Check beam setting
		if beam_threshold != None:
			self.progress_dict["BeamSearch"] = beam_threshold

		#Learn each cycle
		for cycle in self.progress_dict.keys():
			if isinstance(cycle, int):
			
				if self.progress_dict[cycle]["State"] == "Complete":
					print("\t Cycle " + str(cycle) + " already complete.")
					
				#This cycle is not yet finished
				else:

					#-----------------#
					#BACKGROUND STAGE
					#-----------------#
					if self.progress_dict[cycle]["Background_State"] != "Complete":
						
						#Check if ngram extraction is finished
						if self.progress_dict[cycle]["Background_State"] == "None":
							check_files = self.Load.list_output(type = "ngrams")
							pop_list = []
							for i in range(len(self.progress_dict[cycle]["Background"])):
								if self.progress_dict[cycle]["Background"][i] + "." + self.nickname + ".ngrams.p" in check_files:
									pop_list.append(i)

							#Pop items separately in reverse order
							if len(pop_list) > 0:
								for i in sorted(pop_list, reverse = True):
									self.progress_dict[cycle]["Background"].pop(i)
							
							#If remaining background files, process them
							if len(self.progress_dict[cycle]["Background"]) > 0:
								print("\tNow processing remaining files: " + str(len(self.progress_dict[cycle]["Background"])))
								self.Association.find_ngrams(self.progress_dict[cycle]["Background"], workers)
								
							#Change state
							self.progress_dict[cycle]["Background_State"] = "Ngrams"
							self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
						
						#Check if ngram merging is finished
						if self.progress_dict[cycle]["Background_State"] == "Ngrams":
							files = [filename + "." + self.nickname + ".ngrams.p" for filename in self.data_dict[cycle]["Background"]]
						
							print("\tNow merging ngrams for files: " + str(len(files)))
							ngrams = self.Association.merge_ngrams(files, freq_threshold)
							
							#Save data and state
							self.Load.save_file(ngrams, nickname + ".Cycle-" + str(cycle) + ".Merged-Grams.p")
							self.progress_dict[cycle]["Background_State"] = "Merged"
							self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
						
						#Check if association_dict has been made
						if self.progress_dict[cycle]["Background_State"] == "Merged":
							ngrams = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".Merged-Grams.p")
							association_dict = self.Association.calculate_association(ngrams = ngrams, save = False)
							del ngrams
							self.Load.save_file(association_dict, nickname + ".Cycle-" + str(cycle) + ".Association_Dict.p")
							self.progress_dict[cycle]["Background_State"] = "Complete"
							self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
							self.association_dict = association_dict
							
					else:
						print("\tLoading association_dict.")
						self.association_dict = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".Association_Dict.p")
						
					#-----------------#
					#CANDIDATE STAGE
					#-----------------#	
					
					if self.progress_dict[cycle]["Candidate_State"] != "Complete":

						print("Initializing Candidates module")
						C = Candidates(self.language, self.Load, workers, self.association_dict)
						
						#Find beam search threshold
						if self.progress_dict["BeamSearch"] == "None" or self.progress_dict["BeamSearch"] == {}:
							print("Finding Beam Search settings.")

							delta_threshold, best_candidates = delta_grid_search(candidate_file = self.data_dict["BeamCandidates"], 
																	test_file = self.data_dict["BeamTest"], 
																	workers = workers, 
																	mdl_workers = mdl_workers,
																	association_dict = self.association_dict, 
																	freq_threshold = beam_freq_threshold,
																	language = self.language, 
																	in_dir = self.in_dir, 
																	out_dir = self.out_dir, 
																	s3 = self.s3, 
																	s3_bucket = self.s3_bucket,
																	nickname = self.nickname,
																	max_words = self.max_words,
																	)
							self.progress_dict["BeamSearch"] = delta_threshold
							
							self.progress_dict[cycle]["Candidate_State"] = "Threshold"
							self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)

						#If saved, load beam search threshold
						else:
							print("Loading Beam Search settings.")
							delta_threshold = self.progress_dict["BeamSearch"]
							self.progress_dict[cycle]["Candidate_State"] = "Threshold"

						#For a fixed set experiment, we use the same data so we keep the best candidates
						if fixed_set == []:
						
							#Check which files have been completed
							if self.progress_dict[cycle]["Candidate_State"] == "Threshold":
								check_files = self.Load.list_output(type = "candidates")
								pop_list = []
								for i in range(len(self.progress_dict[cycle]["Candidate"])):
									if self.progress_dict[cycle]["Candidate"][i] + ".candidates.p" in check_files:
										pop_list.append(i)							
										
								#Pop items separately in reverse order
								if len(pop_list) > 0:
									for i in sorted(pop_list, reverse = True):
										self.progress_dict[cycle]["Candidate"].pop(i)
									
								#If remaining candidate files, process them
								if len(self.progress_dict[cycle]["Candidate"]) > 0:
									print("\n\tNow processing remaining files: " + str(len(self.progress_dict[cycle]["Candidate"])))
									
									#Multi-process#
									if workers > len(self.progress_dict[cycle]["Candidate"]):
										candidate_workers = len(self.progress_dict[cycle]["Candidate"])
									else:
										candidate_workers = workers
										
									pool_instance = mp.Pool(processes = candidate_workers, maxtasksperchild = 1)
									distribute_list = [(delta_threshold, x) for x in self.progress_dict[cycle]["Candidate"]]
									pool_instance.map(partial(process_candidates, 
																			association_dict = self.association_dict.copy(),
																			language = self.language,
																			in_dir = self.in_dir,
																			out_dir = self.out_dir,
																			s3 = self.s3, 
																			s3_bucket = self.s3_bucket,
																			mode = "candidates",
																			max_words = self.max_words,
																			), distribute_list, chunksize = 1)
									pool_instance.close()
									pool_instance.join()
									
								self.progress_dict[cycle]["Candidate_State"] = "Merge"
								self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)

							#Merge and Save candidates
							if self.progress_dict[cycle]["Candidate_State"] == "Merge":
								output_files = [filename + ".candidates.p" for filename in self.data_dict[cycle]["Candidate"]]
								candidates = self.Candidates.merge_candidates(output_files, freq_threshold)
							
								self.Load.save_file(candidates, nickname + ".Cycle-" + str(cycle) + ".Candidates.p")
								self.progress_dict[cycle]["Candidate_State"] = "Dict"
								self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
								
							#Make association vectors
							if self.progress_dict[cycle]["Candidate_State"] == "Dict":
								
								candidates = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".Candidates.p")
								candidate_dict = self.Candidates.get_association(candidates, self.association_dict)
								self.Load.save_file(candidate_dict, nickname + ".Cycle-" + str(cycle) + ".Candidate_Dict.p")
								
								self.progress_dict[cycle]["Candidate_State"] == "Complete"
								self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
								
							
							else:
								print("\tLoading candidate_dict.")
								candidate_dict = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".Candidate_Dict.p")
								candidates = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".Candidates.p")
							
							del self.association_dict
					
						#If there was a fixed set of training/testing files
						elif fixed_set != []:

							candidates = best_candidates
							candidate_dict = self.Candidates.get_association(candidates, self.association_dict)
							del self.association_dict
							self.progress_dict[cycle]["Candidate_State"] == "Complete"

					#-----------------#
					#MDL STAGE
					#-----------------#
					if no_mdl == False:
						if self.progress_dict[cycle]["MDL_State"] != "Complete":
						
							#Prep test data for MDL
							if self.progress_dict[cycle]["MDL_State"] == "None":
								MDL = MDL_Learner(self.Load, self.Encode, self.Parse, freq_threshold = 1, vectors = candidate_dict, candidates = candidates)
								MDL.get_mdl_data(self.progress_dict[cycle]["Test"], workers = mdl_workers)
								self.Load.save_file(MDL, nickname + ".Cycle-" + str(cycle) + ".MDL.p")
								
								self.progress_dict[cycle]["MDL_State"] = "EM"
								self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
							
							#Run EM-based Tabu Search
							if self.progress_dict[cycle]["MDL_State"] == "EM":
								
								try:
									MDL.search_em(turn_limit, mdl_workers)
								except:
									MDL = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".MDL.p")
									MDL.search_em(turn_limit, mdl_workers)
									
								self.Load.save_file(MDL, nickname + ".Cycle-" + str(cycle) + ".MDL.p")
								self.progress_dict[cycle]["MDL_State"] = "Direct"
								self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
								
							#Run direct Tabu Search
							if self.progress_dict[cycle]["MDL_State"] == "Direct":
								
								try:
									MDL.search_direct(turn_limit*3, mdl_workers)
								except:
									MDL = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".MDL.p")
									MDL.search_direct(turn_limit*3, mdl_workers)
								
								#Get grammar to save
								grammar_dict = defaultdict(dict)
								for i in range(len(MDL.candidates)):
									grammar_dict[i]["Constructions"] = MDL.candidates[i]
									grammar_dict[i]["Matches"] = MDL.matches[i]
										
								#Save grammar
								self.Load.save_file(grammar_dict, nickname + ".Cycle-" + str(cycle) + ".Final_Grammar.p")
								
								self.progress_dict[cycle]["MDL_State"] = "Complete"
								self.progress_dict[cycle]["State"] = "Complete"
								self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)	
								
								del MDL

					elif no_mdl == True:
						print("Calculating MDL")

						self.progress_dict[cycle]["MDL_State"] = "Complete"
						self.progress_dict[cycle]["State"] = "Complete"
				
		#-----------------#
		#MERGING STAGE
		#-----------------#
		if self.progress_dict[cycle]["State"] == "Complete":
			
			if no_mdl == False:
				print("Starting to merge fold grammars.")
				grammar_files = [nickname + ".Cycle-" + str(i) + ".Final_Grammar.p" for i in range(cycles)]
				final_grammar = self.merge_grammars(grammar_files)
				self.Load.save_file(final_grammar, self.language + ".Grammar.p")

			else:
				final_grammar = list(candidates.keys())
				self.Load.save_file(final_grammar, self.nickname + ".Grammar_BeamOnly.p")
				
	#-------------------------------------------------------------------------------
	
	def merge_grammars(self, grammar_files, no_mdl = False):
	
		all_grammars = {}
		
		if no_mdl == False:
			#Load all grammar files
			for file in grammar_files:
			
				current_dict = self.Load.load_file(file)
				
				#Iterate over constructions in current fold grammar
				for key in current_dict.keys():
					current_construction = current_dict[key]["Constructions"]
					current_construction = current_construction.tolist()
					current_matches = current_dict[key]["Matches"]
					
					#Reformat
					new_construction = []
					for unit in current_construction:
						new_type = unit[0]
						new_index = unit[1]
							
						if new_type != 0:
							new_construction.append(tuple((new_type, new_index)))
					
					#Make hashable
					new_construction = tuple(new_construction)
					
					#Add to dictionary
					if new_construction not in all_grammars:
						all_grammars[new_construction] = {}
						all_grammars[new_construction]["Matches"] = current_matches
						all_grammars[new_construction]["Selected"] = 1
					
					else:
						all_grammars[new_construction]["Matches"] += current_matches
						all_grammars[new_construction]["Selected"] += 1
						
			#Done loading grammars
			print("Final grammar for " + self.language + " contains "  + str(len(list(all_grammars.keys()))))
			final_grammar = list(all_grammars.keys())
			final_grammar = self.Parse.format_grammar(final_grammar)

		else:
			final_grammar = []
			for file in grammar_files:
				current_dict = self.Load.load_file(file)
				for key in current_dict:
					if key not in final_grammar:
						final_grammar.append(key)
		
		return final_grammar				
			
	#-------------------------------------------------------------------------------
	
	def divide_data(self, cycles, cycle_size, fixed_set = []):
		
		data_dict = defaultdict(dict)

		#For a fixed set experiment, we use the same data for all simulations
		if fixed_set != []:

			data_dict["BeamCandidates"] = fixed_set
			data_dict["BeamTest"] = fixed_set
			
			for cycle in range(cycles):
				data_dict[cycle]["Test"] = fixed_set
				data_dict[cycle]["Candidate"] = fixed_set
				data_dict[cycle]["Background"] = fixed_set

		#Otherwise we get unique data
		else:

			input_files = self.Load.list_input()		
			
			#Get number of files to use for each purpose
			num_test_files = cycle_size[0]
			num_candidate_files = cycle_size[1]
			num_background_files = cycle_size[2]
			num_cycle_files = cycle_size[0] + cycle_size[1] + cycle_size[2]
			
			#Get Beam Search tuning files
			candidate_i = random.randint(0, len(input_files))
			candidate_file = input_files.pop(candidate_i)
			
			test_i = random.randint(0, len(input_files))
			test_file = input_files.pop(test_i)
			
			#Get and divide input data
			data_dict["BeamCandidates"] = candidate_file
			data_dict["BeamTest"] = test_file
			
				
			#Get unique data for each cycle
			for cycle in range(cycles):
					
				#Randomize remaining files
				random.shuffle(input_files)
				cycle_files = []
					
				#Gather as many files as required
				for segment in range(num_cycle_files):
					current_file = input_files.pop()
					cycle_files.append(current_file)
						
				#Assign files as final MDL test data
				random.shuffle(cycle_files)
				test_files = []
				for file in range(num_test_files):
					current_file = cycle_files.pop()
					test_files.append(current_file)
				data_dict[cycle]["Test"] = test_files
					
				#Assign files as candidate estimation data
				random.shuffle(cycle_files)
				candidate_files = []
				for file in range(num_candidate_files):
					current_file = cycle_files.pop()
					candidate_files.append(current_file)
				data_dict[cycle]["Candidate"] = candidate_files
					
				#Assign files as candidate estimation data
				random.shuffle(cycle_files)
				background_files = []
				for file in range(num_background_files):
					current_file = cycle_files.pop()
					background_files.append(current_file)
				data_dict[cycle]["Background"] = background_files
			
		return data_dict
		
	#-------------------------------------------------------------------------------
	
	def set_progress(self):
	
		progress_dict = defaultdict(dict)
		progress_dict["BeamSearch"] = "None"
		
		for cycle in self.data_dict.keys():
			if isinstance(cycle, int):
				
				progress_dict[cycle]["State"] = "Incomplete"
				progress_dict[cycle]["Background"] = self.data_dict[cycle]["Background"].copy()
				progress_dict[cycle]["Background_State"] = "None"
				progress_dict[cycle]["Candidate"] = self.data_dict[cycle]["Candidate"].copy()
				progress_dict[cycle]["Candidate_State"] = "None"
				progress_dict[cycle]["Test"] = self.data_dict[cycle]["Test"].copy()
				progress_dict[cycle]["MDL_State"] = "None"
			
		return progress_dict
	
