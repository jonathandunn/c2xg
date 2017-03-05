#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

#-- Helper function to create the hypothesis space for a given representation type

def process_get_candidates(filename, 
								Parameters, 
								Grammar, 
								expand_check = True,
								file_extension = ".Candidate.Default",
								annotation_types = "",
								max_candidate_length = "",
								frequency_threshold_perfile = "",
								initial_flag = True
								):

	import time
	import os.path
	from process_input.pandas_open import pandas_open
	from process_input.get_frequencies import get_frequencies
	from constituent_reduction.expand_sentences import expand_sentences
	from candidate_extraction.create_templates import create_templates
	from candidate_extraction.read_candidates import read_candidates
	from candidate_extraction.write_candidates import write_candidates
	from candidate_extraction.templates_to_candidates import templates_to_candidates
	from association_measures.get_phrase_count import get_phrase_count
	
	from process_input.get_temp_filename import get_temp_filename
	output_file = get_temp_filename(filename, file_extension, candidate_flag = True)
	
	if os.path.isfile(output_file) and initial_flag == False:
	
		print("\t\tFile already exists, no need to remake: " + str(output_file))
	
	else:
	
		start_beginning = time.time()
		
		if annotation_types == "":
			annotation_types = Parameters.Annotation_Types
		
		if max_candidate_length == "":
			max_candidate_length = Parameters.Max_Candidate_Length
		
		if frequency_threshold_perfile == "":
			frequency_threshold_perfile = Parameters.Frequency_Threshold_Perfile

		#---------------------------------------------------------------------------------------------#
		#3: Ingest input files and create DataFrames of index values representing sentences ----------#
		#---------------------------------------------------------------------------------------------#

		print("")
		print("\t\tIngesting input files.")

		current_df = pandas_open(filename, Parameters, Grammar, write_output = False)
		lemma_frequency, pos_frequency, category_frequency, number_of_words = get_frequencies(current_df, Grammar)
		
		#---------------------------------------------------------------------------------------------#
		#4: Create expanded sentences with recursive material reduced --------------------------------#
		#---------------------------------------------------------------------------------------------#

		if expand_check == True:
		
			print("")
			print("\t\tExpanding sentences to reduce recursive structures")
			print("")
			current_df = expand_sentences(current_df, Grammar, write_output = False)
			
			#Get frequency of reduced phrases#
			lemma_frequency, pos_frequency = get_phrase_count(current_df, Grammar, lemma_frequency, pos_frequency)
		
		else:
			#Modify current_df to match expanded format#
			
			if len(current_df) > 3:
			
				current_df.loc[:,"Alt"] = 0
				current_df = current_df.loc[:,['Sent', 'Alt', 'Mas', "Lex", 'Pos', 'Cat']]
			
		#---------------------------------------------------------------------------------------------#
		#5: Extract candidate constructions from linguistic expressions ------------------------------#
		#---------------------------------------------------------------------------------------------#
		
		if len(current_df) > 3:
			sequence_list = create_templates(annotation_types, max_candidate_length)
			candidate_dictionary = templates_to_candidates(current_df, 
															filename, 
															sequence_list, 
															annotation_types, 
															max_candidate_length, 
															frequency_threshold_perfile
															)

			#Write full candidates for further use#
			final_dictionary = {}
			
			final_dictionary['lemma_frequency'] = lemma_frequency
			final_dictionary['pos_frequency'] = pos_frequency
			final_dictionary['category_frequency'] = category_frequency
			final_dictionary['number_of_words'] = number_of_words
			final_dictionary['candidate_dictionary'] = candidate_dictionary
			final_dictionary['sequence_list'] = sequence_list
			final_dictionary['Grammar'] = Grammar
			
		else:
			
			final_dictionary = {}
			print("Empty candidates. Not writing file")

		write_candidates(output_file, final_dictionary)	
		
		print("")
		end_beginning = time.time()
		print("\t\tTotal time for " + str(output_file) + ": " + str(end_beginning - start_beginning))
	
	return
#---------------------------------------------------------------------------------------------------#

def get_candidates(Parameters, Grammar, expand_check, file_extension, run_parameter = 0):

	#Note: annotation_types, max_candidate_length, frequency threshold may differ from parameters#
	#Parameter value taken as default if not specified#
						
	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1
		
		print("")
		print("Starting C2xG.Get_Candidates")
		print("")
		
		import datetime
		import time
		import sys
		import multiprocessing as mp
		import pandas as pd
		from functools import partial
		import cytoolz as ct
		
		from get_candidates import process_get_candidates
		from candidate_extraction.read_candidates import read_candidates
		from process_input.annotate_files import annotate_files
		
		#Start multi-processing for file processing#
		pool_instance=mp.Pool(processes = Parameters.CPUs_General, maxtasksperchild = None)
		pool_instance.map(partial(process_get_candidates, 
										Parameters = Parameters,
										Grammar = Grammar,
										expand_check = expand_check,
										file_extension = file_extension,
										annotation_types = "",
										max_candidate_length = "",
										frequency_threshold_perfile = ""
										), Parameters.Input_Files, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for file processing#
		
	return
#----------------------------------------------------------------------------------------------------#		