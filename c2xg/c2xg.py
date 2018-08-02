import os
import random
import pickle
from collections import defaultdict

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

class C2xG(object):

	#-------------------------------------------------------------------------------
	
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
		
	#-------------------------------------------------------------------------------
	
	def eval_mdl(self, files, workers):
	
		print("Initiating MDL evaluation: " + str(files))
		
		for file in files:
			print("\tStarting " + file)
			MDL = MDL_Learner(self.Load, self.Encode, self.Parse, freq_threshold = 10, vectors = {"na": 0}, candidates = self.model)
			MDL.get_mdl_data([file], workers = workers, learn_flag = False)
			current_mdl = MDL.evaluate_subset(subset = False)
	
		
	#-------------------------------------------------------------------------------
		
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
		
	def learn(self, nickname, cycles = 1, cycle_size = (1, 5, 20), ngram_range = (3,6), freq_threshold = 10, turn_limit = 10, workers = 1):
	
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
			
		else:
			print("Initializing learning state.")
			self.data_dict = self.divide_data(cycles, cycle_size)
			self.progress_dict = self.set_progress()
			self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
			
		#Learn each cycle
		for cycle in self.progress_dict.keys():
			
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
						ngrams = self.Association.merge_ngrams(files)
						
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
						
				else:
					print("\tLoading association_dict.")
					association_dict = self.Load.load_file(nickname + ".Cycle-" + str(cycle) + ".Association_Dict.p")
					
				#-----------------#
				#CANDIDATE STAGE
				#-----------------#	
				if self.progress_dict[cycle]["Candidate_State"] != "Complete":
				
					#Check which files have been completed
					if self.progress_dict[cycle]["Candidate_State"] == "None":
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
							self.Candidates.find(ngrams = ngram_range,
													workers = workers,
													files = self.progress_dict[cycle]["Candidate"]
													)
																
						self.progress_dict[cycle]["Candidate_State"] = "Merge"
						self.Load.save_file((self.progress_dict, self.data_dict), self.model_state_file)
					
					#Merage and Save candidates
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
						MDL = MDL_Learner(self.Load, self.Encode, self.Parse, freq_threshold = freq_threshold, vectors = candidate_dict, candidates = candidates)
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
				
		#Get number of files to use for each purpose
		num_test_files = cycle_size[0]
		num_candidate_files = cycle_size[1]
		num_background_files = cycle_size[2]
		num_cycle_files = cycle_size[0] + cycle_size[1] + cycle_size[2]
		
		#Get and divide input data
		data_dict = defaultdict(dict)
		input_files = self.Load.list_input()
			
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
		
		for cycle in self.data_dict.keys():
		
			progress_dict[cycle]["State"] = "Incomplete"
			progress_dict[cycle]["Background"] = self.data_dict[cycle]["Background"].copy()
			progress_dict[cycle]["Background_State"] = "None"
			progress_dict[cycle]["Candidate"] = self.data_dict[cycle]["Candidate"].copy()
			progress_dict[cycle]["Candidate_State"] = "None"
			progress_dict[cycle]["Test"] = self.data_dict[cycle]["Test"].copy()
			progress_dict[cycle]["MDL_State"] = "None"
			
		return progress_dict
	