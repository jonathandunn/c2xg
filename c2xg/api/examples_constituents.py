#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

# This script takes a model for autonomous extraction and tests the coverage of that model on #
#--- a supplied text. ------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------#
def process_examples_constituents(input_file, Parameters, Grammar):

	import time 
	import csv
	import pandas
	from process_input.pandas_open import pandas_open
	from feature_extraction.process_extraction import process_extraction
	from feature_extraction.get_coverage_column import get_coverage_column
	from constituent_reduction.process_learned_constituents import process_learned_constituents
	import cytoolz as ct
	import codecs
	
	start_beginning = time.time()
	
	#INGEST TEST FILE ----------------------------------------------------------------------------#
	print("Ingesting input files.")
	input_dataframe = pandas_open(input_file, 
									Parameters,
									Grammar,
									save_words = True,
									write_output = False,
									delete_temp = False
									)

	#EXPAND TEST FILE ----------------------------------------------------------------------------#
	examples_file = input_file + ".Examples"
	
	print("")
	print("Savings constituent identifications to file: head-first.") 
	total_match_df_lr, full_removed_dictionary_lr, dependence_dictionary_lr, counter = process_learned_constituents(input_dataframe, 
																					Grammar.POS_List, 
																					Grammar.Lemma_List, 
																					Grammar.Constituent_Dict[0], 
																					"LR", 
																					"PRINT", 
																					0,
																					Parameters.Encoding_Type, 
																					examples_file + ".Head-First.txt"
																					)
	
	print("Savings constituent identifications to file: head-last.") 
	total_match_df_rl, full_removed_dictionary_rl, dependency_dictionary_rl, counter = process_learned_constituents(input_dataframe, 
																					Grammar.POS_List, 
																					Grammar.Lemma_List, 
																					Grammar.Constituent_Dict[1], 
																					"RL", 
																					"PRINT", 
																					0,
																					Parameters.Encoding_Type, 
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

def examples_constituents(Parameters, run_parameter = 0):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		#Run parameter keeps pool workers out for this imported module#
		run_parameter = 1

		import datetime
		import sys
		import multiprocessing as mp
		from functools import partial
		
		from api.examples_constituents import process_examples_constituents

		#Import required script-specific modules#
		from candidate_extraction.read_candidates import read_candidates
		from constituent_reduction.expand_sentences import expand_sentences
		from feature_extraction.process_extraction import process_extraction
		from process_input.annotate_files import annotate_files
		
		#LOAD DATA FROM MODEL FILE -------------------------------------------------------------------#

		print("Loading model file.")
		try:
			Grammar = read_candidates(Parameters.Data_File_Constituents)
		except:
			print("Getting constituent examples requires a constituent model!")
			sys.kill()

		#1: Annotate plain text input files  ---------------------------------------------------------#

		if Parameters.Run_Tagger == True:
		
			conll_files = []
			
			for input_file in Parameters.Input_Files:
		
				conll_files += annotate_files(input_file, Parameters, Grammar)
			
			Parameters.Input_Files = conll_files
			input_files = conll_files
			
			#Only need to run tagger once#
			Parameters.Run_Tagger = False
		
		#Get input files if tagger not run#
		else:
			input_files = Parameters.Input_Files
		#----------------------------------------------------------------------------------------------#
	
		#Now, multi-process for input files#
		pool_instance=mp.Pool(processes = Parameters.CPUs_General, maxtasksperchild = None)
		pool_instance.map(partial(process_examples_constituents, Parameters = Parameters, Grammar = Grammar), input_files, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for input files#
		
	return
#-------------------------------------------------------------------------------------------#