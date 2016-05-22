#-----C2xG, v 1.0 ----------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#---- Copyright, 2015-2016 Jonathan E. Dunn --------------------------------------------------#
#---------- www.jdunn.name -------------------------------------------------------------------#
#---------- jonathan.edwin.dunn@gmail.com ----------------------------------------------------#
#---------- Illinois Institute of Technology, Department of Computer Science -----------------#
#---------------------------------------------------------------------------------------------#
#---This file specifies the parameters and settings needed for running the scripts -----------#
#----from the command line; variable descriptions apply to function-calls as well ------------#
#---------------------------------------------------------------------------------------------#

nickname = "TEST"			#Name this series of experiments and the produced models#

#1. POS-TAGGING AND INPUT---------------------------------------------------------------------#

nlp_system = "rdrpos"		#Which dependency to use for pos-tagging. Usually RDRPOSTagger is preferred.#
language = "English"		#Language is used to select the correct RDRPOSTagger model; thus, specify tagset is necessary.#
docs_per_file = 50050		#Number of documents from the raw input file to include in each CoNLL formatted file. For balancing CPU / memory in multi-processing#

#Stanford CoreNLP Parameters: Used only for Chinese / Arabic support#
memory_limit = "4g"									#How much memory to start Jar with#
working_directory = "./files_data/pos_stanford/"	#Location of StanfordNLP dependencies#
pos_model = "NAME.tagger"							#Which Stanford POS model to use#	

encoding_type = "utf-8"								#Encoding type to use after loading files; best to use utf-8#
delete_temp = True									#True / False: Delete temp files during processing; does not include Candidate files output from learn_candidates#

#Items with these pos-tags will not be included in the unit index lists and thus will be discarded#
illegal_pos = ['$', '""', '(', ')', ':', '``', '-rrb-', '-lrb-', '.', 'sym', '--', 'sym', 'url', "''", ',', "'"]

#--------------------------------------------------------------------------------------------#

#2. LEARNING PARAMETERS: PHRASE STRUCTURE, CANDIDATES, CONSTRUCTIONS, AND USAGE -------------#

#2A. Phrase Structure#
phrase_structure_ngram_length = 2			#Maximum range of n-grams to use for learning phrase structure rules#
significance = 0.15							#T-Test significance level for determining head-status#
independence_threshold = 5000				#Number of adjacent occurrences to consider a head independent#
constituent_threshold = 1					#Formulated as [Mean + (Std.Dev * threshold)]#

#2B. Candidates#
frequency_threshold_individual = 100				#Individual units (words, pos-tags, semantic categories) below this threshold will be assigned to 0 index#
frequency_threshold_constructions = 100				#Frequency threshold for candidates, AFTER merging#
frequency_threshold_constructions_perfile = 10		#Frequency threshold for candidates within each file; if low, huge numbers of candidates. However, no candidates below file threshold in a file will be added to total threshold# 
annotation_types = ['Pos']			#Representations to use when searching for candidates. By default, ['Lem', 'Pos', 'Cat']; however, representations can be removed as desired#
max_construction_length = 5							#Max units in candidates. This is an exhaustive search, so larger numbers quickly multiply the search space (e.g., 5 = 1,024 sequences and 6 = 4,096 sequences#

#2C. Constructions#
pairwise_threshold_lr = 1	#Formulated as [Mean + (Std.Dev * threshold)]#
pairwise_threshold_rl = 1	#Formulated as [Mean + (Std.Dev * threshold)]#

#2D. Feature Extraction#
use_centroid = True			#Do or do not use centroid normalization for producing feature vectors. This is essentially finding expected frequencies for features#
use_metadata = True			#Metadata should be in the format: "Field:Value,Field:Value,Field:Value [\t] Text" Feature extraction w/meta-data requires annotate_files_check == Yes# If not extracting vectors, this will trigger removal of meta-data from input files
relative_freq = True		#Relative frequencies if True, otherwise Raw frequencies; overpowered by use_centroid which produces probabilities
full_scope = True			#If True, use full grammar; if False use lexical (e.g., for baseline)
#--------------------------------------------------------------------------------------------#

#3. PROCESSING ------------------------------------------------------------------------------#
debug = True						#Will save debugging info (like phrase structure rules) to debug folder#
number_of_cpus_annotate = 10			#Number of CPUs to use for part-of-speech annotation#
number_of_cpus_prepare = 10			#Number of CPUs to use for learning phrase structure rules#
number_of_cpus_processing = 10		#Number of CPUs to use for learning candidates#
number_of_cpus_pruning = 10			#Number of CPUs to use for learning constructions#
number_of_cpus_extract = 10			#Number of CPUs to use for evaluation, extraction, and annotation#
#--------------------------------------------------------------------------------------------#

#INPUT / OUTPUT FILES AND LOCATIONS ---------------------------------------------------------#
root_location = "../../../../data/"				#Relative path to data storage from the "c2xg/c2xg" folder#

#Strictly for reading#
semantic_dictionary_file = "./files_data/dictionaries/Dictionary.English.ukWac.100.txt"		#Path to desired semantic dictionary#
emoji_file = "./files_data/emojis/Emoji.List.txt"											#Path to list of emojis; helpful for dealing with noisy web-crawled data#

annotate_pos = True		#True: input raw tex, one document per line; False: input CoNLL formatted texts#
use_metadata = True		#Metadata should be in the format: "Field:Value,Field:Value,Field:Value [\t] Text" 
						#Feature extraction w/meta-data requires annotate_pos == Yes#
						#Meta-data needs to be at least categorical (integers) in order to allow sparse vectors in pandas. Use variable identifiers#

#If input files are unannotated, turn on annotate_pos above#
input_files = [

				]

#Output files are the files produced by learn_candidates and needed for learn_constructions#
#The purpose of this is to control which files are used for learning ----------------------#

candidate_files = [

					]
					
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#BEGIN VARIABLE DEFINITIONS THAT SHOULDN'T BE CHANGED-----------------------------------------#
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

input_folder = root_location + "Input"
temp_folder = input_folder + "/Temp"
debug_folder = input_folder + "/Debug"
output_folder = root_location + "Output"
examples_directory = output_folder

data_file_constituents = output_folder + "/" + nickname + ".1.Constituents.model"
data_file_constructions = output_folder + "/" + nickname + ".2.Constructions.model"
data_file_usage = output_folder + "/" + nickname + ".3.Usage.model"

data_file_vectors = output_folder + "/" + nickname + ".Association.Vectors"

settings_dictionary = {}
settings_dictionary['memory_limit'] = memory_limit
settings_dictionary['working_directory'] = working_directory
settings_dictionary['pos_model'] = pos_model
settings_dictionary['language'] = language
settings_dictionary['nlp_system'] = nlp_system

output_suffix = nickname + ".FreqIndv=" + str(frequency_threshold_individual) + ".FreqCon=" + str(frequency_threshold_constructions) + ".Length=" + str(max_construction_length)
output_file = output_folder + "/" + nickname + ".Associations.FreqIndv=" + str(frequency_threshold_individual) + ".FreqCon=" + str(frequency_threshold_constructions) + ".Length=" + str(max_construction_length) + ".csv"
output_file_pruned = output_file.replace(".csv", "") + ".Pruned.csv"

data_file_readable = debug_folder + "/Debug.Readable Corpus." + output_suffix + ".txt"
data_file_reductions = debug_folder + "/Debug.Reductions." + output_suffix + ".txt"
 
#DEBUG SETTINGS AND FILES#
debug_file = debug_folder + "/Debug." + output_suffix + "."

output_files = []
for file in candidate_files:
	output_files.append(str(temp_folder + "/" + file))

#---------------------------------------------------------------------------------------------#
#These should be set to 1 for best results ---------------------------------------------------#
#---------------------------------------------------------------------------------------------#
phrase_ngram_threshold = 1
phrase_ngram_threshold_perfile = 1

phrase_ngram_const_threshold = 1
phrase_ngram_const_threshold_perfile = 1
#---------------------------------------------------------------------------------------------#