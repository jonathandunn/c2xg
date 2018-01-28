#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

def learn_c2xg(Parameters):
			
	from api.learn_c2xg import learn_c2xg
	learn_c2xg(Parameters)
		
	return
		
def learn_rdrpos_model(Parameters):

	from api.learn_rdrpos_model import learn_rdrpos_model
	learn_rdrpos_model(Parameters)
		
	return
		
def learn_dictionary(Parameters):
		
	from api.learn_dictionary import learn_dictionary
	learn_dictionary(Parameters)
		
	return
		
def annotate_pos(Parameters):
		
	from api.annotate_pos import annotate_pos
	annotate_pos(Parameters)
		
	return
	
def get_indexes(Parameters):	
		
	from api.get_indexes import get_indexes
	get_indexes(Parameters)
		
	return		
	
def learn_usage(Parameters):		
		
	from api.learn_usage import learn_usage
	learn_usage(Parameters)
		
	return
	
def learn_association(Parameters):		
		
	from api.learn_association import learn_association
	learn_association(Parameters)
		
	return
	
def learn_idioms(Parameters):
		
	from api.learn_idioms import learn_idioms
	learn_idioms(Parameters)
		
	return
	
def learn_constituents(Parameters):
		
	from api.learn_constituents import learn_constituents
	learn_constituents(Parameters)
		
	return

def learn_constructions(Parameters):		
		
	from api.learn_constructions import learn_constructions
	learn_constructions(Parameters)
		
	return
	

def get_vectors(Parameters):		
		
	from api.get_vectors import get_vectors
	get_vectors(Parameters)
		
	return
	
def evaluate_constructions(Parameters, eval_type):		
		
	from api.evaluate_constructions import evaluate_constructions
	evaluate_constructions(Parameters, eval_type)
		
	return
	
def examples_constituents(Parameters):		
		
	from api.examples_constituents import examples_constituents
	examples_constituents(Parameters)
		
	return

	
class Grammar:

	def __init__(self):
		
		self.Type = "Unlearned"
		self.Idiom_List = []
		
		return

		
class Parameters:

	def __init__(self, parameter_file = ""):
		
		import os.path
		import os

		self.Nickname = "Nickname"
		self.POS_Tagger = "rdrpos"
		self.Language = "English"
		self.Lines_Per_File = 500000
		self.Dictionary_File = "Dictionary.English.ukWac.100.txt"
		self.Encoding_Type = "utf-8"
		self.Delete_Temp = False
		self.Illegal_POS = []
		self.Distance_Threshold = 0.0000001
		self.Freq_Threshold_Individual = 1
		self.Freq_Threshold_Candidates = 1
		self.Freq_Threshold_Candidates_Perfile = 1
		self.Annotation_Types = ["Lex", 'Pos', 'Cat']
		self.Max_Candidate_Length_MWEs = 2
		self.Max_Candidate_Length_Constituents = 2
		self.Max_Candidate_Length_Constructions = 2
		self.Coverage_Threshold = 0.01
		self.Use_Centroid = False
		self.Debug = True
		self.CPUs_Annotate = 1
		self.CPUs_Prepare = 1
		self.CPUs_Processing = 1
		self.CPUs_Merging = 1
		self.CPUs_Pruning = 1
		self.CPUs_Extract = 1
		self.Root_Location = os.path.join("..", "..", "..", "..", "data")
		self.Emoji_File = "Emoji.List.txt"
		self.Run_Tagger = False
		self.Use_Metadata = False
		self.Training_Format = "DF"
		self.Input_Files = []
		self.Candidate_Files = []
		self.Training_Files = []
		
		if parameter_file == "":
			print("Loading blank parameters.")
		
		elif os.path.isfile(parameter_file + ".py"):
			
			print("Loading " + parameter_file + ".py")
			self.initialize(parameter_file)
		
		else:
			print("ERROR: Specified parameter file doesn't exist: " + parameter_file)
			
		return		

	def initialize(self, parameter_file, gui_flag = False):
		
		import os
		
		from modules.process_input import check_folders
		from modules.process_input import create_category_dictionary
		from modules.process_input import create_emoji_dictionary
		from modules.process_input import create_category_index
		
		print("\tLoading parameters from file.")
		
		try:
			if gui_flag == False:
				import importlib
				pm = importlib.import_module(parameter_file)
			
			elif gui_flag == True:
				from importlib.machinery import SourceFileLoader
				pm = SourceFileLoader("set_parameters", parameter_file).load_module()

		except ImportError:
			print("Error in specified parameters file.")
			sys.kill()
			
		#Load, set, and initialize parameters#
		pm.set_parameters(self)
		
		self.Input_Folder = os.path.join(self.Root_Location, "Input")
		self.Dict_Folder = os.path.join(self.Input_Folder, "Dict_Files")
		self.POS_Training_Folder = os.path.join(self.Input_Folder, "POS_Training")
		self.POS_Testing_Folder = os.path.join(self.Input_Folder, "POS_Testing")
		self.Temp_Folder = os.path.join(self.Input_Folder, "Temp")
		self.Candidate_Folder = os.path.join(self.Temp_Folder, "Candidates")
		self.Debug_Folder = os.path.join(self.Input_Folder, "Debug")
		self.Output_Folder = os.path.join(self.Root_Location, "Output")
		self.Examples_Directory = self.Output_Folder
		self.Parameters_Folder = os.path.join(self.Root_Location, "Parameters")
		
		print("\tLoading semantic dictionary from file.")
		self.Dictionary_File = os.path.join(".", "files_data", "dictionaries", self.Dictionary_File)
		self.Semantic_Category_Dictionary = create_category_dictionary(self.Dictionary_File, self.Encoding_Type)
		self.Category_List = create_category_index(self.Semantic_Category_Dictionary)
		
		print("\tLoading emoji list from file.")
		self.Emoji_File = os.path.join(".", "files_data", "emojis", self.Emoji_File)
		self.Emoji_Dictionary = create_emoji_dictionary(self.Emoji_File)
		
		print("\tChecking and creating necessary folders.")
		check_folders(self.Input_Folder, 
						self.Temp_Folder,
						self.Candidate_Folder,
						self.Debug_Folder, 
						self.Output_Folder, 
						self.Dict_Folder, 
						self.POS_Training_Folder, 
						self.POS_Testing_Folder,
						self.Parameters_Folder
						)

		self.Data_File_Indexes = os.path.join(self.Output_Folder, self.Nickname + ".0.Indexes.model")
		self.Data_File_Idioms = os.path.join(self.Output_Folder, self.Nickname + ".1.Idioms.model")
		self.Data_File_Constituents = os.path.join(self.Output_Folder, self.Nickname + ".2.Constituents.model")
		self.Data_File_Constructions = os.path.join(self.Output_Folder, self.Nickname + ".3.Constructions.model")
		self.Data_File_Usage = os.path.join(self.Output_Folder, self.Nickname + ".4.Usage.model")
		self.Data_File_Vectors = os.path.join(self.Output_Folder, self.Nickname + ".Association.Vectors")

		self.Output_Suffix = self.Nickname + ".FreqIndv=" + str(self.Freq_Threshold_Individual) + ".Length=" + str(self.Max_Candidate_Length_Constructions)
		self.Output_File = os.path.join(self.Output_Folder, self.Nickname + ".Associations.FreqIndv=" + str(self.Freq_Threshold_Individual) + ".Length=" + str(self.Max_Candidate_Length_Constructions) + ".csv")
		self.Output_File_Pruned = self.Output_File.replace(".csv", "") + ".Pruned.csv"

		self.Data_File_Readable = os.path.join(self.Debug_Folder, "Debug.Readable Corpus." + self.Output_Suffix + ".txt")
		self.Data_File_Reductions = os.path.join(self.Debug_Folder, "Debug.Reductions." + self.Output_Suffix + ".txt")
		self.Debug_File = os.path.join(self.Debug_Folder, "Debug." + self.Output_Suffix + ".")
		
		import os
		
		self.Output_Files = []
		for file in self.Candidate_Files:
			self.Output_Files.append(os.path.join(self.Temp_Folder, file))
			
		if self.Run_Tagger == False:
			self.Input_Files = [os.path.join(self.Input_Folder, "Temp", file) for file in self.Input_Files]
			
		print("Finished loading parameters.")

		return