import os
import random
import pickle
import copy
import operator
from collections import defaultdict
import multiprocessing as mp
from functools import partial

#Depending on usage, may be importing from the same package
#try:
from modules.Encoder import Encoder
from modules.Loader import Loader
from modules.Parser import Parser
from modules.Association import Association
from modules.Candidates import Candidates
from modules.MDL_Learner import MDL_Learner

#Or from idNet package
# except:

	# from c2xg.modules.Encoder import Encoder
	# from c2xg.modules.Loader import Loader
	# from c2xg.modules.Parser import Parser
	# from c2xg.modules.Association import Association
	# from c2xg.modules.Candidates import Candidates
	# from c2xg.modules.MDL_Learner import MDL_Learner
	# os.chdir(os.path.join(".", "c2xg"))
	
#------------------------------------------------------------

def eval_mdl(files, workers, candidates, Load, Encode, Parse, report = False):
	
	print("Initiating MDL evaluation: " + str(files))
		
	for file in files:
		print("\tStarting " + file)			
		MDL = MDL_Learner(Load, Encode, Parse, freq_threshold = -1, vectors = {"na": 0}, candidates = candidates)
		MDL.get_mdl_data([file], workers = workers, learn_flag = False)
		current_mdl = MDL.evaluate_subset(subset = False)
			
	if report == True:
		return current_mdl
#------------------------------------------------------------		

def delta_grid_search(candidate_file, test_file, workers, association_dict, language, in_dir, out_dir, s3, s3_bucket):
	
	print("\nStarting grid search for beam search settings.")
	result_dict = {}
		
	delta_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 1.0]
	
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
								in_dir = in_dir,
								out_dir = out_dir,
								s3 = s3, 
								s3_bucket = s3_bucket
								), distribute_list, chunksize = 1)
	pool_instance.close()
	pool_instance.join()
				
	#Now MDL
	if language == "zho":
		zho_split = True
	else:
		zho_split = False
		
	Load = Loader(in_dir, out_dir, language, s3, s3_bucket)
	Encode = Encoder(Loader = Load, zho_split = zho_split)
	Parse = Parser(Load, Encode)
	
	for threshold in delta_thresholds:
		print("\tStarting MDL search for " + str(threshold))
		filename = str(candidate_file + ".delta." + str(threshold) + ".p")
		candidates = Load.load_file(filename)
		
		if len(candidates) < 5:
			print("\tNot enough candidates!")
		
		else:		
			mdl_score = eval_mdl(files = [test_file], candidates = candidates, workers = workers, Load = Load, Encode = Encode, Parse = Parse, report = True)
			result_dict[threshold] = mdl_score
			print("\tThreshold: " + str(threshold) + " and MDL: " + str(mdl_score))
		
	#Get threshold with best score
	print(result_dict)
	best = min(result_dict.items(), key=operator.itemgetter(1))[0]
		
	return best

#------------------------------------------------------------

def process_candidates(input_tuple, association_dict, language, in_dir, out_dir, s3, s3_bucket, mode = ""):

	threshold =  input_tuple[0]
	candidate_file = input_tuple[1]
	
	print("\tStarting " + str(threshold))
	Load = Loader(in_dir, out_dir, language, s3, s3_bucket)
	C = Candidates(language = language, Loader = Load, association_dict = association_dict)
	
	if mode == "candidates":
		filename = str(candidate_file + ".candidates.p")
		
	else:
		filename = str(candidate_file) + ".delta." + str(threshold) + ".p"
	
	if filename not in Load.list_output():
	
		candidates = C.process_file(candidate_file, threshold, save = False)
		Load.save_file(candidates, filename)
	
	#Clean
	del association_dict
	del C
	
	return

#-------------------------------------------------------------------------------

class C2xG(object):
	
	def __init__(self, data_dir, language, s3 = False, s3_bucket = "", nickname = "", model = "", zho_split = False):
	
		#Initialize
		in_dir = os.path.join(data_dir, "IN")
		
		if nickname == "":
			out_dir = os.path.join(data_dir, "OUT")
		else:
			out_dir = os.path.join(data_dir, "OUT", nickname)
			
		self.language = language
		self.zho_split = zho_split
		self.Load = Loader(in_dir, out_dir, language = self.language, s3 = s3, s3_bucket = s3_bucket)
		self.Encode = Encoder(Loader = self.Load, zho_split = self.zho_split)
		self.Association = Association(Loader = self.Load)
		self.Candidates = Candidates(language = self.language, Loader = self.Load)
		self.Parse = Parser(self.Load, self.Encode)
		
		self.in_dir = in_dir
		self.out_dir = out_dir
		self.s3 = s3
		self.s3_bucket = s3_bucket

		#Try to load default or specified model
		if model == "":
			model = self.language + ".Grammar.p"
		
		try:
			modelname = os.path.join(".", "data", "models", model)
			with open(modelname, "rb") as handle:
				self.model = pickle.load(handle)
		except:
			try:
				modelname = os.path.join("..", "c2xg", "c2xg", "data", "models", model)
				with open(modelname, "rb") as handle:
					self.model = pickle.load(handle)

			except:
				print("No model exists, loading empty model.")
				self.model = None
			
		self.n_features = len(self.model)
		
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
			lines = self.Parse.parse_idNet(input, self.model, workers)
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
		
	def learn(self, nickname, cycles = 1, cycle_size = (1, 5, 20), ngram_range = (3,6), freq_threshold = 10, turn_limit = 10, workers = 1, states = None):
	
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
			self.data_dict = self.divide_data(cycles, cycle_size)
			self.progress_dict = self.set_progress()
			self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
			
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
								if self.progress_dict[cycle]["Background"][i] + ".ngrams.p" in check_files:
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
							files = [filename + ".ngrams.p" for filename in self.data_dict[cycle]["Background"]]
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

							delta_threshold = delta_grid_search(candidate_file = self.data_dict["BeamCandidates"], 
																	test_file = self.data_dict["BeamTest"], 
																	workers = workers, 
																	association_dict = self.association_dict, 
																	language = self.language, 
																	in_dir = self.in_dir, 
																	out_dir = self.out_dir, 
																	s3 = self.s3, 
																	s3_bucket = self.s3_bucket
																	)
							self.progress_dict["BeamSearch"] = delta_threshold
							
							self.progress_dict[cycle]["Candidate_State"] = "Threshold"
							self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
							
						
						#If saved, load beam search threshold
						else:
							print("Loading Beam Search settings.")
							delta_threshold = self.progress_dict["BeamSearch"]
							self.progress_dict[cycle]["Candidate_State"] = "Threshold"
						
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
																		mode = "candidates"
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
							candidate_dict = self.Candidates.get_association(candidates, association_dict)
							self.Load.save_file(candidate_dict, nickname + ".Cycle-" + str(cycle) + ".Candidate_Dict.p")
							
							self.progress_dict[cycle]["Candidate_State"] == "Complete"
							self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
							
						
					else:
						print("\tLoading candidate_dict.")
						candidate_dict = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".Candidate_Dict.p")
						candidates = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".Candidates.p")
					
					del association_dict
					#-----------------#
					#MDL STAGE
					#-----------------#
					if self.progress_dict[cycle]["MDL_State"] != "Complete":
					
						#Prep test data for MDL
						if self.progress_dict[cycle]["MDL_State"] == "None":
							MDL = MDL_Learner(self.Load, self.Encode, self.Parse, freq_threshold = 0, vectors = candidate_dict, candidates = candidates)
							MDL.get_mdl_data(self.progress_dict[cycle]["Test"], workers = workers)
							self.Load.save_file(MDL, nickname + ".Cycle-" + str(cycle) + ".MDL.p")
							
							self.progress_dict[cycle]["MDL_State"] = "EM"
							self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
						
						#Run EM-based Tabu Search
						if self.progress_dict[cycle]["MDL_State"] == "EM":
							
							MDL = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".MDL.p")
							MDL.search_em(turn_limit, workers)
							self.Load.save_file(MDL, nickname + ".Cycle-" + str(cycle) + ".MDL.p")
							
							self.progress_dict[cycle]["MDL_State"] = "Direct"
							self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
							
						#Run direct Tabu Search
						if self.progress_dict[cycle]["MDL_State"] == "Direct":
							MDL = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".MDL.p")
							MDL.search_direct(turn_limit*3, workers)
							
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
				
		#-----------------#
		#MERGING STAGE
		#-----------------#
		if self.progress_dict[cycle]["State"] == "Complete":
			
			print("Starting to merge fold grammars.")
			grammar_files = [nickname + ".Cycle-" + str(i) + ".Final_Grammar.p" for i in range(cycles)]
			final_grammar = self.merge_grammars(grammar_files)
			self.Load.save_file(final_grammar, self.language + ".Grammar.p")
				
	#-------------------------------------------------------------------------------
	
	def merge_grammars(self, grammar_files):
	
		all_grammars = {}
		
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
		
		return final_grammar				
			
	#-------------------------------------------------------------------------------
	
	def divide_data(self, cycles, cycle_size):
		
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
		data_dict = defaultdict(dict)
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
	