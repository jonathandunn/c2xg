#-----C2xG, v 1.0 ----------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#---- Copyright, 2015-2016 Jonathan E. Dunn --------------------------------------------------#
#---------- www.jdunn.name -------------------------------------------------------------------#
#---------- jonathan.edwin.dunn@gmail.com ----------------------------------------------------#
#---------- Illinois Institute of Technology, Department of Computer Science -----------------#
#---------------------------------------------------------------------------------------------#
# This script takes a model for autonomous extraction and tests the coverage of that model on #
#--- a supplied text. ------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------#
def process_examples_constructions(input_file, 
									input_folder, 
									output_folder, 
									candidate_list, 
									lemma_list, 
									pos_list, 
									word_list, 
									category_list, 
									semantic_category_dictionary, 
									sequence_list, 
									max_construction_length, 
									annotation_types, 
									pruned_vector_dataframe, 
									encoding_type, 
									phrase_constituent_list, 
									lemma_dictionary, 
									pos_dictionary, 
									category_dictionary, 
									examples_directory, 
									):
#-------------------------------------------------------------------------------------------------#

	import time 
	import csv
	import pandas
	from functions_input.pandas_open import pandas_open
	from functions_constituent_reduction.expand_sentences import expand_sentences
	from functions_autonomous_extraction.process_extraction import process_extraction
	from functions_autonomous_extraction.get_coverage_column import get_coverage_column
	from functions_input.get_temp_filename import get_temp_filename
	
	start_beginning = time.time()
	
	#---------------------------------------------------------------------------------------------#
	#INGEST TEST FILE ----------------------------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#

	print("")
	print("Ingesting input files.")
	input_dataframe = pandas_open(input_file, 
									encoding_type, 
									semantic_category_dictionary, 
									word_list, 
									lemma_list, 
									pos_list, 
									lemma_dictionary, 
									pos_dictionary, 
									category_dictionary,
									write_output = False
									)
	print("Finished ingesting input files.")

	#---------------------------------------------------------------------------------------------#
	#EXPAND TEST FILE ----------------------------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#

	print("")
	print("Expanding sentences to reduce recursive structures.") 
	input_dataframe = expand_sentences(input_dataframe, 
										lemma_list, 
										pos_list, 
										category_list, 
										encoding_type, 
										write_output = False, 
										phrase_constituent_list = phrase_constituent_list, 
										)
	print("Finished expanding sentences.")
	print("")

	full_sentences = input_dataframe.loc[:,'Sent'].tolist()
	number_of_sentences = len(set(full_sentences))

	#---------------------------------------------------------------------------------------------#
	#EXTRACT FEATURES ----------------------------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#
	#Get examples filename and overwrite if exists#
	write_examples = get_temp_filename(input_file, ".Examples")
	fo = open(write_examples, "w")
	fo.close()
	#---------------------------------------------#
	
	process_extraction(candidate_list, 
						max_construction_length, 
						input_dataframe, 
						lemma_list, 
						pos_list, 
						category_list, 
						number_of_sentences,
						full_scope = False,
						write_examples = write_examples
							)


	#---------------------------------------------------------------------------------------------#
	#PRINT TIME ELAPSED --------------------------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#
	print("")
	end_beginning = time.time()
	print("Total time for " + str(input_file) + " is " + str(end_beginning - start_beginning))
	
	return
#------------------------------------------------------------------------------------------------#

def examples_constituents(model_file, 
							encoding_type, 
							number_processes, 
							input_folder, 
							output_folder, 
							examples_directory, 
							annotate_pos, 
							settings_dictionary,
							input_files,
							run_parameter = 0
							):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1
		#---------------------------------------------------------------------------------------------#
		#IMPORT DEPENDENCIES -------------------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#

		import datetime
		import sys
		import multiprocessing as mp
		from functools import partial
		
		from examples_constructions import process_examples_constructions

		#Import required script-specific modules#
		from functions_candidate_extraction.read_candidates import read_candidates
		from functions_constituent_reduction.expand_sentences import expand_sentences
		from functions_autonomous_extraction.process_extraction import process_extraction
		from functions_annotate.annotate_files import annotate_files

		#---------------------------------------------------------------------------------------------#
		#LOAD DATA FROM MODEL FILE -------------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#

		print("Loading model file.")
		write_dictionary = read_candidates(model_file)

		candidate_list = write_dictionary['candidate_list']
		lemma_list = write_dictionary['lemma_list']
		pos_list = write_dictionary['pos_list']
		word_list = write_dictionary['word_list']
		category_list = write_dictionary['category_list']
		semantic_category_dictionary = write_dictionary['semantic_category_dictionary']
		sequence_list = write_dictionary['sequence_list']
		max_construction_length = write_dictionary['max_construction_length']
		annotation_types = write_dictionary['annotation_types']
		pruned_vector_dataframe = write_dictionary['pruned_vector_dataframe']
		encoding_type = encoding_type
		phrase_constituent_list = write_dictionary['phrase_constituent_list']
		lemma_dictionary = write_dictionary['lemma_dictionary']
		pos_dictionary = write_dictionary['pos_dictionary']
		category_dictionary = write_dictionary['category_dictionary']
		emoji_dictionary = write_dictionary['emoji_dictionary']

		#Fix elements which throw off ARFF importing#
		try:
			comma_index = lemma_list.index(",")
			lemma_list[comma_index] = "<COMMA>"

		except:
			comma_index = "n/a"
	
		try:
			quote_index = lemma_list.index('"')
			lemma_list[quote_index] = "<QUOTE>"

		except:
			quote_index = "n/a"

		del write_dictionary
		print("Finished loading model file.")
		
		#---------------------------------------------------------------------------------------------#
		#1: Annotate plain text input files  ---------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#
		if annotate_pos == True:
		
			updated_input = []
			
			for input_file in input_files:
			
				output_name = input_file + ".Vectors"
				
				conll_file = annotate_files(input_folder, 
											input_file, 
											settings_dictionary, 
											encoding_type, 
											number_processes, 
											emoji_dictionary, 
											docs_per_file = 100000000000
											)
						
				updated_input += conll_file
				
		elif annotate_pos == False:
		
			updated_input = []
	
			for filename in input_files:
				input_short = filename
				updated_input.append(input_folder + "/" + filename)
			
			input_files = updated_input

		input_files = updated_input
		#----------------------------------------------------------------------------------------------#
	
		#Now, multi-process for input files#
		pool_instance=mp.Pool(processes = number_processes, maxtasksperchild = None)
		pool_instance.map(partial(process_examples_constructions, 
									input_folder=input_folder, 
									output_folder=output_folder, 
									candidate_list=candidate_list, 
									lemma_list=lemma_list, 
									pos_list=pos_list, 
									word_list=word_list, 
									category_list=category_list, 
									semantic_category_dictionary=semantic_category_dictionary, 
									sequence_list=sequence_list, 
									max_construction_length=max_construction_length, 
									annotation_types=annotation_types, 
									pruned_vector_dataframe=pruned_vector_dataframe, 
									encoding_type=encoding_type, 
									phrase_constituent_list=phrase_constituent_list, 
									lemma_dictionary=lemma_dictionary, 
									pos_dictionary=pos_dictionary, 
									category_dictionary=category_dictionary, 
									examples_directory=examples_directory
									), input_files, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for input files#
		
	return
#----------------------------------------------------------------------------------------#

#Prevent pool workers from starting here#
if __name__ == '__main__':
#---------------------------------------#

	#START CODE FOR RUNNING FROM COMMAND LINE#
	import sys

	#Get parameters file to use#
	parameters_file = str(sys.argv[1])

	#Import parameter values and global variables#
	#All parameter and global variables must be prefaced with "pm."  ---#

	import importlib
	import examples_constructions

	try:
		pm = importlib.import_module(parameters_file)
	except ImportError:
		print("Error in specified parameters file. Format is 'files_parameters.FILENAME'")

	examples_constituents(pm.data_file_constructions, 
							pm.encoding_type, 
							pm.number_of_cpus_extract, 
							pm.input_folder, 
							pm.output_folder, 
							pm.examples_directory, 
							pm.annotate_pos, 
							pm.settings_dictionary,
							pm.input_files
							)

	#END CODE FOR RUNNING FROM COMMAND LINE#