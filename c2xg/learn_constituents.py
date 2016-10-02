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
						distance_threshold,
						data_file_grammar,
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
		from functions_input.process_create_unit_index import process_create_unit_index
		from functions_input.create_category_dictionary import create_category_dictionary
		from functions_input.create_emoji_dictionary import create_emoji_dictionary
		from functions_input.create_category_index import create_category_index
		from functions_input.get_index_lists import get_index_lists

		from functions_phrase_structure.get_matrix import get_matrix
		from functions_phrase_structure.identify_catenae_pairs import identify_catenae_pairs
		from functions_phrase_structure.join_catenae_pairs import join_catenae_pairs
		
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
			input_files = [input_folder + "/Temp/" + file for file in input_files]
		#---------------------------------------------------------------------------------------------#
		#2: Create index of  frequency reduced labels from input files -------------------------------#
		#---------------------------------------------------------------------------------------------#

		print("Loading semantic category dictionary from file.")
		semantic_category_dictionary = create_category_dictionary(semantic_dictionary_file, encoding_type)
		category_list = create_category_index(semantic_category_dictionary)
	
		print("Creating index of frequency reduced labels from input files.")

		#Full list contains lemma, pos, and role lists#
		full_dictionary = process_create_unit_index(input_files, frequency_threshold_individual, encoding_type, semantic_category_dictionary, illegal_pos, number_of_cpus_prepare)
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
		#4: Learn basic phrase structure rules for constituent reduction -------------------------------#
		#---------------------------------------------------------------------------------------------#
	
		print("")
		print("Learning phrase structure rules for constituent reduction.")
		print("")
		
		print("First, get frequency and association (LR and RL) matrices.")
		pair_frequency_dictionary, lr_association_dictionary, rl_association_dictionary, base_frequency_dictionary, file_counter = get_matrix(pos_list, 
																																input_files, 
																																distance_threshold, 
																																number_of_cpus_prepare,
																																encoding_type,
																																semantic_category_dictionary,
																																word_list,
																																lemma_list,
																																lemma_dictionary,
																																pos_dictionary,
																																category_dictionary,
																																delete_temp
																																)

		print("")
		print("Second, identify catenae-pairs.")
		pair_status_dictionary, pair_head_dictionary, head_status_dictionary = identify_catenae_pairs(pair_frequency_dictionary, 
																										lr_association_dictionary, 
																										rl_association_dictionary,
																										base_frequency_dictionary,
																										pos_list
																										)
		
		
		#Write debug info for pair predictions#
		if debug == True:
		
			fw = open(debug_file + "PairStatus", "w", encoding=encoding_type)
			for pair in pair_status_dictionary:
				fw.write(str(pos_list[pair[0]]))
				fw.write(" : ")
				fw.write(str(pos_list[pair[1]]))
				fw.write(",")
				fw.write(str(pair_status_dictionary[pair]))
				
				if pair_status_dictionary[pair] == "Catenae" and pair[0] != pair[1]:
					fw.write(",")
					fw.write(str(pair_head_dictionary[pair]))
					
				fw.write("\n")
		#Done writing debug info#
		
		print("")
		print("Third, join catenae-pairs into constituents.")
		time_join = time.time()
		
		pair_status_list = [x for x in pair_status_dictionary.keys() if pair_status_dictionary[x] == "Catenae"]
		
		#Multi-process #
		pool_instance=mp.Pool(processes = number_of_cpus_prepare, maxtasksperchild = None)
		rule_list = pool_instance.map(partial(join_catenae_pairs,
												pair_status_list = pair_status_list, 
												head_status_dictionary = head_status_dictionary,
												unit_list = pos_list,
												input_files = input_files[0:file_counter],
												pos_list = pos_list,
												encoding_type = encoding_type,
												semantic_category_dictionary = semantic_category_dictionary,
												word_list = word_list,
												lemma_list = lemma_list,
												lemma_dictionary = lemma_dictionary,
												pos_dictionary = pos_dictionary,
												category_dictionary = category_dictionary,
												delete_temp = delete_temp
												), [i for i in range(len(pos_list))], chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		
		print("Time for joining pairs: " + str(time.time() - time_join))
		rule_list = ct.merge([x for x in rule_list if type(x) is dict])
		
		#Join unit rules as dictionary with rules as key and direction as value#
		rule_dict = {}
		for unit in rule_list.keys():
		
			current_head = unit
			current_direction = head_status_dictionary[current_head]
			
			for current_rule in rule_list[unit]:
					
					rule_dict[current_rule] = current_direction
				
		cfg_dictionary = rule_dict
		
		#Debug info for rules ----------------------------#
		if debug == True:
			fo = open(debug_file + "CFG-Rules", "w")
			for phrase in cfg_dictionary.keys():
				for unit in phrase:
					fo.write(str((pos_list[unit])))
					fo.write(str(" -- "))
				
				fo.write(str("\t"))
				fo.write(str(cfg_dictionary[phrase]))
				fo.write(str("\n"))
				
			fo.close()
		#-------------------------------------------------#
		
		
		#DONE LEARNING, NOW PRESENT AND STORE DATA#
		phrase_constituent_list = reformat_constituents(cfg_dictionary)
			
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
						pm.distance_threshold,
						pm.data_file_constituents,
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