#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

#-- Prerequisite function for creating categorical indexes used to represent lexical items, semantic domains, pos-tags

def get_indexes(Parameters, Grammar = None, input_files = None, Idiom_check = False, run_parameter = 0):

	print("")
	print("Starting C2xG.Get_Indexes")
	print("")
	
	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1

		import time
		import sys
		import multiprocessing as mp
		import pandas as pd
		from functools import partial
		import cytoolz as ct

		#Import required script-specific modules#
		import c2xg
		from process_input.annotate_files import annotate_files
		from process_input.process_create_unit_index import process_create_unit_index
		from process_input.get_index_lists import get_index_lists
		from candidate_extraction.read_candidates import read_candidates
		from candidate_extraction.write_candidates import write_candidates
		
		start_beginning = time.time()
		
		if input_files == None:
			input_files = Parameters.Input_Files

		#---------------------------------------------------------------------------------------------#
		#1: Annotate plain text input files  ---------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#
		if Parameters.Run_Tagger == True and mwe_check == False:
		
			conll_files = []
			
			for input_file in input_files:
		
				conll_files += annotate_files(input_file, Parameters, Grammar = None, metadata = False)
			
			Parameters.Input_Files = conll_files
			Parameters.Run_Tagger = False
		#---------------------------------------------------------------------------------------------#
		#2: Create index of  frequency reduced labels from input files -------------------------------#
		#---------------------------------------------------------------------------------------------#
		print("Creating index of frequency reduced labels from input files.")
		
		#Full list contains lemma, pos, and role lists#
		full_dictionary = process_create_unit_index(Parameters, input_files)
		full_dictionary = get_index_lists(full_dictionary)

		#Separate the various items from the container dictionary#
		lemma_list = full_dictionary['lemma_list']
		pos_list = full_dictionary['pos_list']
		word_list = full_dictionary['word_list']
		category_list = full_dictionary['category_list']
	
		lemma_frequency = full_dictionary['lemma_frequency']
		pos_frequency = full_dictionary['pos_frequency']
		word_frequency = full_dictionary['word_frequency']
		category_frequency = full_dictionary['category_frequency']
	
		lemma_dictionary = full_dictionary['lemma_dictionary']
		pos_dictionary = full_dictionary['pos_dictionary']
		category_dictionary = full_dictionary['category_dictionary']
	
		del full_dictionary
	
		print("")
		print("Words: " + str(len(word_list)))
		print("Lemmas: " + str(len(lemma_list)))
		print("POS: " + str(len(pos_list)))	
		print("Categories: " + str(len(category_list)))
		print("")

		#If debug is on, write files for unit label inventories#
		if Parameters.Debug == True:
		
			from process_input.write_debug import write_debug
			write_debug(category_frequency, 
							lemma_frequency, 
							word_frequency, 
							pos_frequency, 
							Parameters.Debug_File, 
							Parameters.Encoding_Type
							)
			
					
		#---------------------------------------------------------------------------------------------#
		#Write data to single file -------------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#
		if Grammar == None:
			Grammar = c2xg.Grammar()
	
		Grammar.Type = "Indexes"
		Grammar.POS_List = pos_list
		Grammar.Lemma_List = lemma_list
		Grammar.Category_List = category_list
		Grammar.Word_List = word_list
		Grammar.Lemma_Dictionary = lemma_dictionary
		Grammar.POS_Dictionary = pos_dictionary
		Grammar.Category_Dictionary = category_dictionary
		Grammar.Emoji_Dictionary = Parameters.Emoji_Dictionary
		Grammar.Semantic_Category_Dictionary = Parameters.Semantic_Category_Dictionary
	
		print("")
		print("Writing grammar file.")
		write_candidates(Parameters.Data_File_Indexes, Grammar)
		
		print("")
		end_beginning = time.time()
		print("Total time: " + str(end_beginning - start_beginning))
		
	return Grammar
#----------------------------------------------------------------------------------------------------#