#-----C2xG, v 1.0 ----------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#---- Copyright, 2015-2016 Jonathan E. Dunn --------------------------------------------------#
#---------- www.jdunn.name -------------------------------------------------------------------#
#---------- jonathan.edwin.dunn@gmail.com ----------------------------------------------------#
#---------- Illinois Institute of Technology, Department of Computer Science -----------------#
#---------------------------------------------------------------------------------------------#
#-Takes a single chunk of the corpus and the saved constituent grammar from learn_constituents#
#--- and then processes that chunk through ingestion, expansion, and candidate search. -------#
#--- The output is a file containing all candidates and their frequency, ---------------------#
#--- As well as the unit index lists and frequency dictionaries. -----------------------------#
#---------------------------------------------------------------------------------------------#

def process_learn_candidates(filename, 
								phrase_constituent_list, 
								pos_list, 
								lemma_list, 
								category_list, 
								word_list, 
								semantic_category_dictionary, 
								lemma_dictionary, 
								pos_dictionary, 
								category_dictionary, 
								annotate_pos, 
								encoding_type, 
								input_folder, 
								annotation_types, 
								max_construction_length, 
								frequency_threshold_constructions_perfile, 
								emoji_dictionary,
								delete_temp
								):

	import time
	
	from functions_input.create_unit_index import create_unit_index
	from functions_input.create_category_dictionary import create_category_dictionary
	from functions_input.create_category_index import create_category_index
	from functions_input.check_data_files import check_data_files
	from functions_input.pandas_open import pandas_open
	from functions_input.get_index_lists import get_index_lists
	from functions_input.get_frequencies import get_frequencies
	from functions_constituent_reduction.expand_sentences import expand_sentences
	from functions_constituent_reduction.write_reduction_list import write_reduction_list
	from functions_candidate_extraction.create_templates import create_templates
	from functions_candidate_extraction.read_candidates import read_candidates
	from functions_candidate_extraction.write_candidates import write_candidates
	from functions_candidate_extraction.templates_to_candidates import templates_to_candidates
	from functions_candidate_extraction.merge_candidates import merge_candidates
	from functions_candidate_extraction.merge_file_results import merge_file_results
	from functions_candidate_extraction.print_full_candidate_info import print_full_candidate_info
	from functions_candidate_evaluation.get_phrase_count import get_phrase_count
	
	start_beginning = time.time()	

	#---------------------------------------------------------------------------------------------#
	#3: Ingest input files and create DataFrames of index values representing sentences ----------#
	#---------------------------------------------------------------------------------------------#

	print("")
	print("Ingesting input files.")

	current_df = pandas_open(filename, 
								encoding_type, 
								semantic_category_dictionary, 
								word_list, 
								lemma_list, 
								pos_list, 
								lemma_dictionary, 
								pos_dictionary, 
								category_dictionary,
								write_output = False,
								delete_temp = delete_temp
								)

	freq_dict = get_frequencies(current_df, lemma_list, pos_list, category_list)
	
	lemma_frequency = freq_dict['lemma_frequency']
	pos_frequency = freq_dict['pos_frequency']
	category_frequency = freq_dict['category_frequency']
	number_of_words = freq_dict['number_of_words']
	
	del freq_dict	

	#---------------------------------------------------------------------------------------------#
	#4: Create expanded sentences with recursive material reduced --------------------------------#
	#---------------------------------------------------------------------------------------------#

	print("")
	print("Expanding sentences to reduce recursive structures")
	print("")
	
	current_df = expand_sentences(current_df, 
									lemma_list, 
									pos_list, 
									category_list, 
									encoding_type, 
									write_output = False, 
									phrase_constituent_list = phrase_constituent_list
									)
	
	#Get frequency of reduced phrases#
	return_dictionary = get_phrase_count(lemma_list, lemma_frequency, pos_list, pos_frequency, current_df)
	
	lemma_frequency = return_dictionary['lemma_frequency']
	pos_frequency = return_dictionary['pos_frequency']
	
	#---------------------------------------------------------------------------------------------#
	#5: Extract candidate constructions from linguistic expressions ------------------------------#
	#---------------------------------------------------------------------------------------------#
		
	sequence_list = create_templates(annotation_types, max_construction_length)
	candidate_dictionary = templates_to_candidates(current_df, 
													filename, 
													sequence_list, 
													annotation_types, 
													max_construction_length, 
													frequency_threshold_constructions_perfile
													)

	#Write full candidates for further use#
	from functions_input.get_temp_filename import get_temp_filename
	output_file = get_temp_filename(filename, ".Candidates")
	
	final_dictionary = {}
	
	final_dictionary['lemma_frequency'] = lemma_frequency
	final_dictionary['pos_frequency'] = pos_frequency
	final_dictionary['category_frequency'] = category_frequency
	final_dictionary['number_of_words'] = number_of_words
	final_dictionary['pos_list'] = pos_list
	final_dictionary['lemma_list'] = lemma_list
	final_dictionary['category_list'] = category_list
	final_dictionary['word_list'] = word_list
	final_dictionary['phrase_constituent_list'] = phrase_constituent_list
	final_dictionary['semantic_category_dictionary'] = semantic_category_dictionary
	final_dictionary['lemma_dictionary'] = lemma_dictionary
	final_dictionary['pos_dictionary'] = pos_dictionary
	final_dictionary['category_dictionary'] = category_dictionary
	final_dictionary['emoji_dictionary'] = emoji_dictionary
	final_dictionary['candidate_dictionary'] = candidate_dictionary
	final_dictionary['sequence_list'] = sequence_list
	
	write_candidates(output_file, final_dictionary)		
	
	print("")
	end_beginning = time.time()
	print("Total time for " + str(output_file) + ": " + str(end_beginning - start_beginning))
	
	return
#---------------------------------------------------------------------------------------------------#

def learn_candidates(input_files,
						input_folder, 
						output_folder, 
						data_file_grammar, 
						max_construction_length, 
						frequency_threshold_constructions_perfile, 
						number_of_cpus_processing = 1, 
						annotate_pos = False,
						encoding_type = "utf-8", 
						annotation_types = ["Lem", "Pos", "Cat"], 
						settings_dictionary = {},
						docs_per_file = 99999999999,
						delete_temp = False,
						run_parameter = 0
						):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1
		
		import datetime
		import time
		import sys
		import multiprocessing as mp
		import pandas as pd
		from functools import partial
		import cytoolz as ct
		
		from learn_candidates import process_learn_candidates

		#Import required script-specific modules#
		from functions_input.check_folders import check_folders
		from functions_candidate_extraction.read_candidates import read_candidates
		from functions_annotate.annotate_files import annotate_files
	
		#Check if folders exist. If not, create them.#	
		check_folders(input_folder, input_folder + "/Temp/", input_folder + "/Debug/", output_folder)

		#---------------------------------------------------------------------------------------------#
		#DONE WITH IMPORT DEPENDENCIES ---------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#
	
		print("Reading grammar file.")
		final_dictionary = read_candidates(data_file_grammar)
	
		phrase_constituent_list = final_dictionary['phrase_constituent_list']
		pos_list = final_dictionary['pos_list']
		lemma_list = final_dictionary['lemma_list']
		category_list = final_dictionary['category_list']
		word_list = final_dictionary['word_list']
		semantic_category_dictionary = final_dictionary['semantic_category_dictionary']
		lemma_dictionary = final_dictionary['lemma_dictionary']
		pos_dictionary = final_dictionary['pos_dictionary']
		category_dictionary = final_dictionary['category_dictionary']
		emoji_dictionary = final_dictionary['emoji_dictionary']
		
		del final_dictionary
		
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
												number_of_cpus_processing, 
												emoji_dictionary, 
												docs_per_file
												)
			
			input_files = conll_files
			
		else:
			conll_files = []
			for file in input_files:
				conll_files.append(input_folder + "/Temp/" + file)
			input_files = conll_files
		#----------------------------------------------------------------------------------------------#
	
		#Start multi-processing for file processing#
		pool_instance=mp.Pool(processes = number_of_cpus_processing, maxtasksperchild = None)
		pool_instance.map(partial(process_learn_candidates, 
										phrase_constituent_list=phrase_constituent_list, 
										pos_list=pos_list, 
										lemma_list=lemma_list, 
										category_list=category_list, 
										word_list=word_list, 
										semantic_category_dictionary=semantic_category_dictionary, 
										lemma_dictionary=lemma_dictionary, 
										pos_dictionary=pos_dictionary, 
										category_dictionary=category_dictionary, 
										annotate_pos=annotate_pos, 
										encoding_type=encoding_type, 
										input_folder=input_folder, 
										annotation_types=annotation_types, 
										max_construction_length=max_construction_length, 
										frequency_threshold_constructions_perfile=frequency_threshold_constructions_perfile, 
										emoji_dictionary=emoji_dictionary,
										delete_temp=delete_temp
										), input_files, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for file processing#
		
	return
#----------------------------------------------------------------------------------------------------#


#Prevent pool workers from starting here#
if __name__ == '__main__':
#---------------------------------------#
	
	#CODE FOR RUNNING FROM COMMAND LINE#
	import sys

	#Get parameters file to use#
	parameters_file = str(sys.argv[1])

	#Import parameter values and global variables#
	#All parameter and global variables must be prefaced with "pm."  ---#

	import importlib
	from learn_candidates import learn_candidates
	from learn_candidates import process_learn_candidates

	try:
		pm = importlib.import_module(parameters_file)
	except ImportError:
		print("Error in specified parameters file. Format is 'files_parameters.FILENAME'")
		
	learn_candidates(pm.input_files,
						pm.input_folder, 
						pm.output_folder, 
						pm.data_file_constituents, 
						pm.max_construction_length, 
						pm.frequency_threshold_constructions_perfile, 
						pm.number_of_cpus_processing, 
						pm.annotate_pos,
						pm.encoding_type, 
						pm.annotation_types, 
						pm.settings_dictionary,
						pm.docs_per_file,
						pm.delete_temp
					)
		
	#END CODE FOR RUNNING FROM COMMAND LINE#			