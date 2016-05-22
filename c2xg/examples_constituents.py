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
def process_examples_constituents(input_file, 
									input_folder, 
									output_folder, 
									lemma_list, 
									lemma_dictionary, 
									pos_list, 
									pos_dictionary, 
									word_list, 
									category_list, 
									category_dictionary, 
									semantic_category_dictionary, 
									punctuation_breaks_clauses, 
									phrase_constituent_list, 
									examples_directory, 
									annotate_pos, 
									encoding_type
									):
#-------------------------------------------------------------------------------------------------#

	import time 
	import csv
	import pandas
	from functions_input.pandas_open import pandas_open
	from functions_autonomous_extraction.process_extraction import process_extraction
	from functions_autonomous_extraction.get_coverage_column import get_coverage_column
	from functions_constituent_reduction.process_learned_constituents import process_learned_constituents
	import cytoolz as ct
	import codecs
	
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
	examples_file = input_file + ".Examples"
	
	print("")
	print("Savings constituent identifications to file: head-first.") 
	total_match_df_lr, full_removed_dictionary_lr, dependence_dictionary_lr, counter = process_learned_constituents(input_dataframe, 
																					pos_list, 
																					lemma_list, 
																					phrase_constituent_list[0], 
																					"LR", 
																					"PRINT", 
																					0,
																					encoding_type, 
																					examples_file + ".Head-First.txt"
																					)
	
	print("Savings constituent identifications to file: head-last.") 
	total_match_df_rl, full_removed_dictionary_rl, dependency_dictionary_rl, counter = process_learned_constituents(input_dataframe, 
																					pos_list, 
																					lemma_list, 
																					phrase_constituent_list[1], 
																					"RL", 
																					"PRINT", 
																					0,
																					encoding_type, 
																					examples_file + ".Head-Last.txt"
																					)

	full_removed_dictionary = ct.merge(full_removed_dictionary_lr, full_removed_dictionary_rl)
	
	start_list = list(full_removed_dictionary.keys())
	start_list = sorted(start_list, reverse=False)
	
	#Fully schematic example#
	fw = codecs.open(examples_file+".All.txt", "w", encoding = encoding_type)
	
	for start_index in start_list:
		fw.write(str("["))
		temp_df = input_dataframe.query("Mas == @start_index", parser='pandas', engine='numexpr')
		temp_word = temp_df.loc[:,'Word'].values
		fw.write(str(temp_word))
		fw.write(str(" "))
		
		current_length = full_removed_dictionary[start_index]
		print(full_removed_dictionary)
		print(current_length)
		
		if current_length > 1:
			for i in range(1,current_length):
				current_index = start_index + i
				temp_df = input_dataframe.query("Mas == @current_index", parser='pandas', engine='numexpr')
				temp_word = temp_df.loc[:,'Word'].values
				fw.write(str(temp_word))
				fw.write(str(" "))
				
		fw.write(str("]\n"))
		
	fw.close()

	#---------------------------------------------------------------------------------------------#
	#PRINT TIME ELAPSED --------------------------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#
	print("")
	end_beginning = time.time()
	print("Total time for " + str(input_file) + " is " + str(end_beginning - start_beginning))
	
	return
#------------------------------------------------------------------------------------------------#

def examples_constituents(number_processes, 
							model_file, 
							encoding_type, 
							punctuation_breaks_clauses, 
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
		#Run parameter keeps pool workers out for this imported module#
		run_parameter = 1

		#---------------------------------------------------------------------------------------------#
		#IMPORT DEPENDENCIES -------------------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#

		import datetime
		import sys
		import multiprocessing as mp
		from functools import partial
		
		from examples_constituents import process_examples_constituents

		#Import required script-specific modules#
		from functions_candidate_extraction.read_candidates import read_candidates
		from functions_input.open_files import open_files
		from functions_constituent_reduction.expand_sentences import expand_sentences
		from functions_autonomous_extraction.process_extraction import process_extraction
		from functions_annotate.annotate_files import annotate_files

		#---------------------------------------------------------------------------------------------#
		#LOAD DATA FROM MODEL FILE -------------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#

		print("Loading model file.")
		write_dictionary = read_candidates(model_file)

		lemma_list = write_dictionary['lemma_list']
		lemma_dictionary = write_dictionary['lemma_dictionary']
		pos_list = write_dictionary['pos_list']
		pos_dictionary = write_dictionary['pos_dictionary']
		word_list = write_dictionary['word_list']
		category_list = write_dictionary['category_list']
		category_dictionary = write_dictionary['category_dictionary']
		semantic_category_dictionary = write_dictionary['semantic_category_dictionary']
		phrase_constituent_list = write_dictionary['phrase_constituent_list']
		emoji_dictionary = write_dictionary['emoji_dictionary']
		encoding_type = encoding_type
		punctuation_breaks_clauses = punctuation_breaks_clauses

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
			
				input_short = input_file
				output_name = output_folder + "/" + input_short + ".Vectors"
				
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
		pool_instance.map(partial(process_examples_constituents, 
									input_folder=input_folder, 
									output_folder=output_folder, 
									lemma_list=lemma_list, 
									lemma_dictionary=lemma_dictionary, 
									pos_list=pos_list, 
									pos_dictionary=pos_dictionary, 
									word_list=word_list, 
									category_list=category_list, 
									category_dictionary=category_dictionary, 
									semantic_category_dictionary=semantic_category_dictionary, 
									punctuation_breaks_clauses=punctuation_breaks_clauses, 
									phrase_constituent_list=phrase_constituent_list, 
									examples_directory=examples_directory, 
									annotate_pos=annotate_pos, 
									encoding_type=encoding_type
									), input_files, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for input files#
		
	return
#-------------------------------------------------------------------------------------------#

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

	from examples_constituents import process_examples_constituents
	from examples_constituents import examples_constituents

	try:
		pm = importlib.import_module(parameters_file)
	except ImportError:
		print("Error in specified parameters file. Format is 'files_parameters.FILENAME'")

	examples_constituents(pm.number_of_cpus_extract, 
								pm.data_file_model, 
								pm.encoding_type, 
								pm.punctuation_breaks_clauses, 
								pm.input_folder, 
								pm.output_folder, 
								pm.examples_directory, 
								pm.annotate_pos, 
								pm.settings_dictionary,
								pm.input_files
								)

	#END CODE FOR RUNNING FROM COMMAND LINE#
		