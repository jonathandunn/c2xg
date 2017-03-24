#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

def set_parameters(C2xG_Parameters):

	#General settings#
	C2xG_Parameters.Nickname = "German.MDL.3-8"				#Name this series of experiments and the produced models
	C2xG_Parameters.Language = "German"						#Language is used to select the correct RDRPOSTagger model; thus, must have the same name as an available model
	C2xG_Parameters.Lines_Per_File = 5000						#Number of lines from the raw input file to include in each CoNLL formatted file. For balancing CPU / memory in multi-processing
	C2xG_Parameters.Encoding_Type = "utf-8"						#Encoding type to use after loading files; best to use utf-8
	C2xG_Parameters.Illegal_POS = ["X", "x", ".", "punct"]		#Items with these pos-tags will be ignored; for most languages we use the Universal POS-Tags and this ignores punctuation and misc
	C2xG_Parameters.Root_Location = "../../../../Data/"			#Relative path to data storage from the "c2xg/c2xg" folder
	C2xG_Parameters.Emoji_File = "Emoji.List.txt"				#Path to list of emojis; helpful for dealing with noisy web-crawled data
	
	C2xG_Parameters.Annotation_Types = ["Lex", "Pos", "Cat"]	#For learn_association, choose types of representation to include for candidate sequences
	
	#Name of pre-existing dictionary; learn a new one with learn_dictionary
	C2xG_Parameters.Dictionary_File = "German.Aranea.DIM=500.SG=1.HS=1.ITER=25.txt"
	
	#Frequency Thresholds#
	C2xG_Parameters.Freq_Threshold_Individual = 2000			#Individual units (words, pos-tags, semantic categories) below this threshold will be assigned to 0 index
	C2xG_Parameters.Freq_Threshold_Idioms = 1000				#Frequency threshold for candidates total
	C2xG_Parameters.Freq_Threshold_Idioms_Perfile = 10			#Frequency threshold for candidates within each file
	C2xG_Parameters.Freq_Threshold_Constituents = 2000			#Frequency threshold for candidates total
	C2xG_Parameters.Freq_Threshold_Constituents_Perfile = 10	#Frequency threshold for candidates within each file
	C2xG_Parameters.Freq_Threshold_Constructions = 1500			#Frequency threshold for candidates total
	C2xG_Parameters.Freq_Threshold_Constructions_Perfile = 10	#Frequency threshold for candidates within each file
	
	#Candidate Size Constraints#
	C2xG_Parameters.Max_Candidate_Length_Idioms = 5				#Max units in candidates. This is an exhaustive search, so larger numbers quickly multiply the search space (e.g., 5 = 1,024 sequences and 6 = 4,096 sequences)
	C2xG_Parameters.Max_Candidate_Length_Constituents = 5
	C2xG_Parameters.Max_Candidate_Length_Constructions = 5
	
	#Tabu Search Parameters#
	C2xG_Parameters.Use_Freq_Weighting = True					#If True, use frequency-weighted association measures as well as unweighted for grammar sampling
	C2xG_Parameters.Tabu_Thresholds_Number = 100				#Number of discrete thresholds to divide continuous parameter space into for tabu search
	C2xG_Parameters.Tabu_Indirect_Move_Number = 25				#Number of moves to check per feature for each turn of the Tabu Search (for 30*n moves each turn)
	C2xG_Parameters.Tabu_Indirect_Move_Size = 5					#Maximum number of parameters that can be changed in a single move of the Tabu Search (out of 30 possible parameters)
	C2xG_Parameters.Tabu_Direct_Move_Number = 200				#Number of moves to check for direct search
	C2xG_Parameters.Tabu_Direct_Move_Size = 100					#Maximum number of parameters that can be changed in a single move of the indirect Tabu Search (in candidates)
	C2xG_Parameters.Tabu_Random_Checks = 2000					#Number of random grammars to use for initialization (before) and validation (after) Tabu Search

	#For learning semantic dictionary with GenSim's word2vec#
	C2xG_Parameters.Dict_Min_Threshold = 500					#Frequency threshold for inclusion in dictionary
	C2xG_Parameters.Dict_Num_Dimensions = 500					#Dimensionality of neural network
	C2xG_Parameters.Dict_Skip_Gram = 1							#0 = CBOW, 1 = Skip-Grams
	C2xG_Parameters.Dict_Num_Clusters = 100						#Arbitrary number of categories in final dictionary
	C2xG_Parameters.Dict_Num_Iterations = 25					#Fixed number of iterations for learning feature weights#C2xG_Parameters.Delete_Temp = False
	
	#Misc. settings for misc. algorithms#
	C2xG_Parameters.Frequency = "Raw"							#Type of frequency measure to use for CxG vectors: "Raw", "Relative", "TFIDF"
	C2xG_Parameters.Vectors = "CxG"								#Type of vectors to extract: "Lexical", "Units", "CxG", "CxG+Units"
	C2xG_Parameters.Expand_Check = False						#If using CxG features, allow constituents to fill slots#
	C2xG_Parameters.Debug = True								#Will save debugging info (like phrase structure rules) to debug folder
	C2xG_Parameters.Run_Tagger = True							#True: input raw tex, one document per line; False: input CoNLL formatted texts
	C2xG_Parameters.Use_Metadata = False						#Metadata should be in the format: "Field:Value,Field:Value,Field:Value [\t] Text" 
																#Feature extraction w/meta-data requires annotate_pos == Yes
																#Meta-data needs to be at least categorical (integers) in order to allow sparse vectors in pandas. Use variable identifiers#
	C2xG_Parameters.Delete_Temp = False							#If True, clean-up temp files after each fold or whenever possible
	
	#Distribution of data across tasks and folds#
	C2xG_Parameters.CV = 2										#Number of folds for cross-fold validation for each iteration
	C2xG_Parameters.Restarts = 2								#Number of restarts for the tabu search
	C2xG_Parameters.Training_Candidates = 1000000				#Number of documents / sentences (ids) to use for finding candidates / association measures
	C2xG_Parameters.Training_Search = 50000					#Number of documents / sentences (ids) to use for training files during restarts
	C2xG_Parameters.Testing = 100000							#Number of documents / sentences (ids) to use for testing files, once per fold

	#Number of processes separated by type to allow balancing CPU / memory across tasks
	C2xG_Parameters.CPUs_General = 6							#Number of CPUs to use for most tasks
	C2xG_Parameters.CPUs_Merging = 6							#Number of CPUs for merging candidates; higher memory use per CPU
	C2xG_Parameters.CPUs_Learning = 6							#Number of CPUs to use for learning constructions; may have higher memory use per CPU

	#FILES: Input_Files = The input data for learning candidates: Raw Text requires Run_Tagger = True ----#
	#Training, Training-Testing, and Testing files are randomly generated from this list for each fold#
	
	C2xG_Parameters.Input_Files = [
	"German.Testing.txt"
			]

	C2xG_Parameters.Candidate_Files = []
							
	return