#-----C2xG, v 1.0 ----------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#---- Copyright, 2015-2016 Jonathan E. Dunn --------------------------------------------------#
#---------- www.jdunn.name -------------------------------------------------------------------#
#---------- jonathan.edwin.dunn@gmail.com ----------------------------------------------------#
#---------- Illinois Institute of Technology, Department of Computer Science -----------------#
#---------------------------------------------------------------------------------------------#
#This script takes input text and produces the following items as a single file: -------------#
#---Unit indexes for each level of representation (lemma, part-of-speech, category, phrases)--#
#---Phrase types and constituents for each phrase head ---------------------------------------#
#---The semantic category dictionary (pre-existing and created via create_semantic_dictionary)#
#---The emoji dictionary for detecting and labelling emojis and symbolic punctuation sequences#
#---------------------------------------------------------------------------------------------#

def learn_constituents(input_folder, 
						output_folder, 
						emoji_file,
						input_files, 
						semantic_dictionary_file, 
						frequency_threshold_individual, 
						illegal_pos, 
						phrase_structure_ngram_length, 
						significance,
						independence_threshold,
						data_file_grammar,
						constituent_threshold,
						annotate_pos = False,
						encoding_type = "utf-8", 
						docs_per_file = 999999999,
						settings_dictionary = {},
						number_of_cpus_annotate = 1, 
						number_of_cpus_prepare = 1,
						delete_temp = False,
						debug = "",
						debug_file = "",
						run_parameter = 0
						):

	#---------------------------------------------------------------------------------------------#
	#IMPORT DEPENDENCIES -------------------------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#

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
		from functions_annotate.annotate_files import annotate_files

		from functions_input.check_folders import check_folders
		from functions_input.create_unit_index import create_unit_index
		from functions_input.create_category_dictionary import create_category_dictionary
		from functions_input.create_emoji_dictionary import create_emoji_dictionary
		from functions_input.create_category_index import create_category_index
		from functions_input.pandas_open import pandas_open
		from functions_input.get_index_lists import get_index_lists

		from functions_phrase_structure.learn_head_directions import learn_head_directions
		from functions_phrase_structure.learn_phrase_constituents import learn_phrase_constituents
		from functions_phrase_structure.learn_head_independence import learn_head_independence
		from functions_phrase_structure.create_direction_list import create_direction_list
		from functions_phrase_structure.update_pos_list import update_pos_list
		from functions_phrase_structure.update_lemma_list import update_lemma_list
		from functions_phrase_structure.reformat_constituents import reformat_constituents

		from functions_candidate_extraction.read_candidates import read_candidates
		from functions_candidate_extraction.write_candidates import write_candidates

		#Check if folders exist. If not, create them.#	
		check_folders(input_folder, input_folder + "/Temp/", input_folder + "/Debug/", output_folder)
		
		#Create emoji dictionary#
		emoji_dictionary = create_emoji_dictionary(emoji_file)
		
		#---------------------------------------------------------------------------------------------#
		#DONE WITH IMPORT DEPENDENCIES ---------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#

		start_beginning = time.time()

		#---------------------------------------------------------------------------------------------#
		#1: Annotate plain text input files  ---------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#
		if annotate_pos == True:
		
			conll_files = []
			
			for input_file in input_files:
		
				conll_files += annotate_files(input_folder, 
												input_file, 
												settings_dictionary, 
												encoding_type, 
												number_of_cpus_annotate, 
												emoji_dictionary, 
												docs_per_file
												)
			
			input_files = conll_files
			
		else:
			conll_files = []
			for file in input_files:
				conll_files.append(input_folder + "/Temp/" + file)
			input_files = conll_files
		#---------------------------------------------------------------------------------------------#
		#2: Create index of  frequency reduced labels from input files -------------------------------#
		#---------------------------------------------------------------------------------------------#

		print("Loading semantic category dictionary from file.")
		semantic_category_dictionary = create_category_dictionary(semantic_dictionary_file, encoding_type)
		category_list = create_category_index(semantic_category_dictionary)
	
		print("Creating index of frequency reduced labels from input files.")

		#Full list contains lemma, pos, and role lists#
		full_dictionary = create_unit_index(input_files, frequency_threshold_individual, encoding_type, semantic_category_dictionary, illegal_pos)
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
		if debug == True:
		
			from functions_input.write_debug import write_debug
			write_debug(category_frequency, lemma_frequency, word_frequency, pos_frequency, debug_file, encoding_type)
			
		#---------------------------------------------------------------------------------------------#
		#3: Ingest input files and create DataFrames of index values representing sentences ----------#
		#--------and contained within larger HDF5 file for efficiency --------------------------------#
		#---------------------------------------------------------------------------------------------#

		print("")
		print("Ingesting input files.")
		
		data_files = []
		
		#Start multi-processing for loading files#
		pool_instance=mp.Pool(processes = number_of_cpus_prepare, maxtasksperchild = None)
		data_files += pool_instance.map(partial(pandas_open, 
									encoding_type = encoding_type,
									semantic_category_dictionary = semantic_category_dictionary,
									word_list = word_list,
									lemma_list = lemma_list,
									pos_list = pos_list,
									lemma_dictionary = lemma_dictionary,
									pos_dictionary = pos_dictionary,
									category_dictionary = category_dictionary,
									save_words = True,
									write_output = True,
									delete_temp = delete_temp
							), input_files, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for loading files#
	
		#---------------------------------------------------------------------------------------------#
		#4: Learn basic phrase structure rules for constituent reduction -------------------------------#
		#---------------------------------------------------------------------------------------------#
	
		print("")
		print("Learning phrase structure rules for constituent reduction.")
		print("")
		
		print("First, determining which POS categories are phrase heads and, if phrase heads, if they are head-first or head-last.")
		
		#Start multi-processing for learning pos head status#
		pool_instance=mp.Pool(processes = number_of_cpus_prepare, maxtasksperchild = None)
		head_direction_list = pool_instance.map(partial(learn_head_directions, 
														data_files = data_files, 
														phrase_structure_ngram_length = phrase_structure_ngram_length, 
														index_list = pos_list, 
														significance = significance
													), pos_list[1:], chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for learning pos head status#
				
		lr_head_list = create_direction_list(head_direction_list, "Head-First", pos_list)
		rl_head_list = create_direction_list(head_direction_list, "Head-Last", pos_list)
		
		print("")
		print("Second, determining for each phrase head its set of possible constituents.")
		
		#Start multi-processing for learning phrase constituents#
		pool_instance=mp.Pool(processes = number_of_cpus_prepare, maxtasksperchild = None)
		phrase_constituent_list = pool_instance.map(partial(learn_phrase_constituents, 
															data_files = data_files, 
															phrase_structure_ngram_length = phrase_structure_ngram_length, 
															index_list = pos_list, 
															lr_head_list = lr_head_list, 
															rl_head_list = rl_head_list,
															constituent_threshold = constituent_threshold
														), head_direction_list, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for learning phrase constituents#
	
		print("")
		print("Third, determining for each phrase head if it can independently form a phrase.")
	
		#Start multi-processing for learning phrase constituents#
		pool_instance=mp.Pool(processes = number_of_cpus_prepare, maxtasksperchild = None)
		phrase_independence_list = pool_instance.map(partial(learn_head_independence, 
															data_files = data_files, 
															index_list = pos_list, 
															lr_head_list = lr_head_list, 
															rl_head_list = rl_head_list,
															independence_threshold = independence_threshold
														), pos_list, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for learning phrase constituents#

		#Write debug file for phrase reduction rules#
		if debug == True:
			
			fdebug = open(debug_file + "PhraseStructure", "w", encoding=encoding_type)
			for phrase_tuple in phrase_constituent_list:
			
				if phrase_tuple != None:
					pos_index = phrase_tuple[0]
					pos_label = pos_list[pos_index]
					pos_constituents = phrase_tuple[1]
								
					for item in pos_constituents:
					
						real_item = eval(item)
						fdebug.write(str(pos_label.upper() + "_PHRASE: "))
					
						for pos in real_item:
							fdebug.write(str(" "))
							fdebug.write(str(pos_list[int(pos)]))
							
						fdebug.write("\n")
			fdebug.close()
		#Done with phrase reduction debug file#
		
		phrase_constituent_list = reformat_constituents(phrase_constituent_list, 
														lr_head_list, 
														rl_head_list, 
														phrase_structure_ngram_length, 
														phrase_independence_list,
														pos_list
														)
			
		#Update lemma and pos indexes to include phrase-types for dependent heads#
		pos_list = update_pos_list(pos_list, phrase_independence_list)
		lemma_list = update_lemma_list(lemma_list, phrase_independence_list)
	
		#---------------------------------------------------------------------------------------------#
		#5: Write data to single file ----------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#
		final_dictionary = {}
	
		final_dictionary['phrase_constituent_list'] = phrase_constituent_list
		final_dictionary['pos_list'] = pos_list
		final_dictionary['lemma_list'] = lemma_list
		final_dictionary['category_list'] = category_list
		final_dictionary['word_list'] = word_list
		final_dictionary['semantic_category_dictionary'] = semantic_category_dictionary
		final_dictionary['lemma_dictionary'] = lemma_dictionary
		final_dictionary['pos_dictionary'] = pos_dictionary
		final_dictionary['category_dictionary'] = category_dictionary
		final_dictionary['emoji_dictionary'] = emoji_dictionary
	
		print("")
		print("Writing grammar file.")
		write_candidates(data_file_grammar, final_dictionary)
		
		if delete_temp == True:
			print("Deleting temp files.")
			from functions_input.check_data_files import check_data_files
			for file in data_files:
				check_data_files(file)			
		
		print("")
		end_beginning = time.time()
		print("Total time: " + str(end_beginning - start_beginning))
		
	return
#----------------------------------------------------------------------------------------------------#

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
	from learn_constituents import learn_constituents

	try:
		pm = importlib.import_module(parameters_file)
	except ImportError:
		print("Error in specified parameters file. Format is 'files_parameters.FILENAME'")
		
	learn_constituents(pm.input_folder, 
						pm.output_folder, 
						pm.emoji_file,
						pm.input_files, 
						pm.semantic_dictionary_file, 
						pm.frequency_threshold_individual, 
						pm.illegal_pos, 
						pm.phrase_structure_ngram_length, 
						pm.significance,
						pm.independence_threshold,
						pm.data_file_constituents,
						pm.constituent_threshold,
						pm.annotate_pos,
						pm.encoding_type,
						pm.docs_per_file,
						pm.settings_dictionary,
						pm.number_of_cpus_annotate, 
						pm.number_of_cpus_prepare,
						pm.delete_temp,
						pm.debug,
						pm.debug_file
						)
#END CODE FOR RUNNING FROM COMMAND LINE#