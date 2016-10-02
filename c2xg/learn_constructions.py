#-----C2xG, v 1.0 ----------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#---- Copyright, 2015-2016 Jonathan E. Dunn --------------------------------------------------#
#---------- www.jdunn.name -------------------------------------------------------------------#
#---------- jonathan.edwin.dunn@gmail.com ----------------------------------------------------#
#---------- Illinois Institute of Technology, Department of Computer Science -----------------#
#---------------------------------------------------------------------------------------------#
#This script takes a series of files with candidate and unit frequencies and -----------------#
#--- combines them to evaluate candidates using association measures. The output is a single -#
#--- file containing all the data needed for autonomous feature extraction. ------------------#

def learn_constructions(input_folder, 
							training_files,
							training_flag,
							output_folder, 
							output_files, 
							max_construction_length, 
							nickname,
							frequency_threshold,
							encoding_type = "utf-8", 
							annotation_types = ["Lem", "Pos", "Cat"], 
							number_of_cpus_pruning = 1,
							number_of_cpus_merging = 1,
							coverage_threshold = 0.00001,
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
		
		import datetime
		import time
		import sys
		import multiprocessing as mp
		import pandas as pd
		from functools import partial
		import cytoolz as ct

		#Import required script-specific modules#
		from functions_input.check_folders import check_folders

		from functions_candidate_extraction.read_candidates import read_candidates
		from functions_candidate_extraction.write_candidates import write_candidates
		from functions_candidate_extraction.create_templates import create_templates

		from functions_candidate_evaluation.get_formatted_candidates import get_formatted_candidates
		from functions_candidate_evaluation.get_phrase_count import get_phrase_count
		from functions_candidate_evaluation.get_pairwise_df import get_pairwise_df
		from functions_candidate_evaluation.get_dictionary import get_dictionary
		from functions_candidate_evaluation.process_pairwise_feature_vector import process_pairwise_feature_vector
		from functions_candidate_evaluation.process_unitwise_feature_vector import process_unitwise_feature_vector
		from functions_candidate_evaluation.get_frequency_dict import get_frequency_dict
		from functions_candidate_evaluation.write_results import write_results
		from functions_candidate_evaluation.merge_output import merge_output
		from functions_candidate_evaluation.get_df_pairwise import get_df_pairwise
		from functions_candidate_evaluation.get_df_unitwise import get_df_unitwise
		
		from functions_candidate_pruning.process_coverage import process_coverage
		from functions_candidate_pruning.merge_coverage_association import merge_coverage_association
		from functions_candidate_pruning.find_optimum_grammar import find_optimum_grammar
		from functions_candidate_pruning.write_results_pruned import write_results_pruned
		from functions_candidate_pruning.write_model import write_model
	
		#Check if folders exist. If not, create them.#	
		check_folders(input_folder, input_folder + "/Temp/", input_folder + "/Debug/", output_folder)
		
		data_file_model = output_folder + "/" + nickname + ".2.Constructions.model" 
		data_file_vectors = output_folder + "/" + nickname + ".Vectors.p"
		data_file_vectors_pruned = output_folder + "/" + nickname + ".Vectors.Pruned.p"

		#---------------------------------------------------------------------------------------------#
		#DONE WITH IMPORT DEPENDENCIES ---------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#

		start_beginning = datetime.datetime.now().time()
	
		#---------------------------------------------------------------------------------------------#
		#1: Merge provided output files and return necessary data ------------------------------------#
		#---------------------------------------------------------------------------------------------#
	
		input_dictionary = merge_output(output_files, frequency_threshold, number_of_cpus_merging)
	
		pos_list = input_dictionary['pos_list']
		lemma_list = input_dictionary['lemma_list']
		category_list = input_dictionary['category_list']
		word_list = input_dictionary['word_list']
		phrase_constituent_list = input_dictionary['phrase_constituent_list']
		semantic_category_dictionary = input_dictionary['semantic_category_dictionary']
		lemma_dictionary = input_dictionary['lemma_dictionary']
		pos_dictionary = input_dictionary['pos_dictionary']
		category_dictionary = input_dictionary['category_dictionary']
		emoji_dictionary = input_dictionary['emoji_dictionary']
	
		lemma_frequency = input_dictionary['lemma_frequency']
		pos_frequency = input_dictionary['pos_frequency']
		category_frequency = input_dictionary['category_frequency']
		number_of_words = input_dictionary['number_of_words']
		candidate_dictionary = input_dictionary['candidate_dictionary']
	
		sequence_list = create_templates(annotation_types, max_construction_length)

		#---------------------------------------------------------------------------------------------#
		#2: Calcualte Delta P and other evaluation measures ------------------------------------------#
		#---------------------------------------------------------------------------------------------#
	
		total_units = number_of_words
		print("Total words represented by merged output files: " + str(total_units))
		print("")
		
		#Get formatted candidates and frequencies#
		temp_list = get_formatted_candidates(candidate_dictionary)
		candidate_list_formatted = temp_list[0]
		candidate_list_all = temp_list[1]
		candidate_list_pairs = temp_list[2]
		
		del temp_list
		del candidate_dictionary

		print("Creating dictionary of A, B, C, D co-occurrence statistics.")
		start_time = time.time()
		#Start multi-processing for creating pairwise A, B, C, D dataframe#
		pool_instance=mp.Pool(processes = number_of_cpus_pruning, maxtasksperchild = None)
		pairwise_list = pool_instance.map(partial(get_pairwise_df, 
														lemma_frequency=lemma_frequency, 
														lemma_list=lemma_list, 
														pos_frequency=pos_frequency, 
														pos_list=pos_list, 
														category_frequency=category_frequency, 
														category_list=category_list, 
														total_units=total_units
														), candidate_list_pairs, chunksize = 10000)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for creating pairwise A, B, C, D dataframe#
		
		pairwise_dictionary = get_dictionary(pairwise_list)
		del pairwise_list

		end_time = time.time()
		print("\tCreate co-occurrence dictionary: " + str(end_time - start_time))
		
		print("")
		print("Begin building vector of pairwise features.")
		start_time = time.time()
		
		#Start multi-processing for candidate evaluation#
		pool_instance=mp.Pool(processes = number_of_cpus_pruning, maxtasksperchild = None)
		unweighted_feature_vector_list = pool_instance.map(partial(process_pairwise_feature_vector, 
															pairwise_dictionary=pairwise_dictionary,
															freq_weighted=False
															), candidate_list_all, chunksize = 10000)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for candidate evaluation#
		
		#Start multi-processing for candidate evaluation#
		pool_instance=mp.Pool(processes = number_of_cpus_pruning, maxtasksperchild = None)
		weighted_feature_vector_list = pool_instance.map(partial(process_pairwise_feature_vector, 
															pairwise_dictionary=pairwise_dictionary,
															freq_weighted=True
															), candidate_list_all, chunksize = 10000)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for candidate evaluation#
	
		unweighted_vector_df = get_df_pairwise(unweighted_feature_vector_list, "Unweighted")
		weighted_vector_df = get_df_pairwise(weighted_feature_vector_list, "Weighted")	

		pairwise_vector_df = pd.merge(unweighted_vector_df, weighted_vector_df, on=['Candidate', 'Frequency'])

		del unweighted_feature_vector_list
		del weighted_feature_vector_list
		del unweighted_vector_df
		del weighted_vector_df
														
		end_time = time.time()
		print("\tCreate pairwise vectors: " + str(end_time - start_time))

		print("")
		print("Begin building vector of unitwise features.")
		start_time = time.time()
		
		print("\tCreate candidate frequency dictionary")
		candidate_frequency_dict = get_frequency_dict(candidate_list_all)
		
		print("\tCalculate End- and Beginning-Divided Association Measures")
		#Start multi-processing for candidate evaluation#
		pool_instance=mp.Pool(processes = number_of_cpus_pruning, maxtasksperchild = None)
		unweighted_unitwise_vector_list = pool_instance.map(partial(process_unitwise_feature_vector, 
															candidate_frequency_dict=candidate_frequency_dict, 
															lemma_frequency=lemma_frequency, 
															lemma_list=lemma_list, 
															pos_frequency=pos_frequency, 
															pos_list=pos_list, 
															category_frequency=category_frequency, 
															category_list=category_list, 
															total_units=total_units,
															freq_weighted=False
															), candidate_list_all, chunksize = 50000)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for candidate evaluation#
		
		#Start multi-processing for candidate evaluation#
		pool_instance=mp.Pool(processes = number_of_cpus_pruning, maxtasksperchild = None)
		weighted_unitwise_vector_list = pool_instance.map(partial(process_unitwise_feature_vector, 
															candidate_frequency_dict=candidate_frequency_dict, 
															lemma_frequency=lemma_frequency, 
															lemma_list=lemma_list, 
															pos_frequency=pos_frequency, 
															pos_list=pos_list, 
															category_frequency=category_frequency, 
															category_list=category_list, 
															total_units=total_units,
															freq_weighted=True
															), candidate_list_all, chunksize = 50000)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for candidate evaluation#
		
		unweighted_unitwise_vector_df = get_df_unitwise(unweighted_unitwise_vector_list, "Unweighted")
		weighted_unitwise_vector_df = get_df_unitwise(weighted_unitwise_vector_list, "Weighted")
		unitwise_vector_df = pd.merge(unweighted_unitwise_vector_df, weighted_unitwise_vector_df, on='Candidate')
		
		del unweighted_unitwise_vector_list
		del weighted_unitwise_vector_list
		del unweighted_unitwise_vector_df
		del weighted_unitwise_vector_df
		
		full_vector_dataframe = pd.merge(pairwise_vector_df, unitwise_vector_df, on='Candidate')
		
		del pairwise_vector_df
		del unitwise_vector_df

		output_file = output_folder + "/" + nickname + ".OutputVectors"
		output_file_pruned = output_folder + "/" + nickname + ".OutputVectorsPruned"
				
		write_results(full_vector_dataframe, 
						lemma_list, 
						pos_list, 
						category_list, 
						output_file,
						encoding_type = encoding_type, 
						)
		
		write_candidates(data_file_vectors, full_vector_dataframe)
				
		end_time = time.time()
		print("\tCreate unitwise vectors: " + str(end_time - start_time))
	
		#Clean memory#
		del candidate_list_all
		del candidate_list_pairs
		del pairwise_dictionary
		del candidate_frequency_dict

		#---------------------------------------------------------------------------------------------#
		#3: Get candidate cover DataFrame ------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#
		
		print("")
		print("Getting coverage information for grammar evaluation.")
		
		#Collect indexes covered per construction#
		#Multi-process below---------------------#
		
		coverage_df = process_coverage(training_files,
												training_flag,
												candidate_list_formatted, 
												max_construction_length, 
												word_list,
												lemma_list, 
												pos_list, 
												category_list,
												lemma_dictionary, 
												pos_dictionary, 
												category_dictionary,
												semantic_category_dictionary,
												phrase_constituent_list,
												encoding_type,
												number_of_cpus_pruning
												)
												
		print("Writing coverage dictionary")
		write_candidates(debug_file + "Coverage", coverage_df)
		
		full_vector_dataframe = merge_coverage_association(coverage_df, full_vector_dataframe, coverage_threshold, training_files)
		
		#Clean up memory#
		del candidate_list_formatted
		del coverage_df
		
		#---------------------------------------------------------------------------------------------#
		#4: Prune candidate constructions ------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#
		print("")
		print("Begin grammar generation and evaluation.")
		print("")

		optimum_grammar = find_optimum_grammar(full_vector_dataframe, number_of_cpus_pruning)
												
		write_candidates(debug_file + "Optimum_Grammar", optimum_grammar)
		
		#Create .model file for autonomous feature extraction#
		write_model(lemma_list, 
						pos_list, 
						word_list, 
						category_list, 
						semantic_category_dictionary, 
						sequence_list, 
						max_construction_length, 
						annotation_types, 
						optimum_grammar, 
						encoding_type, 
						data_file_model, 
						phrase_constituent_list, 
						lemma_dictionary, 
						pos_dictionary, 
						category_dictionary, 
						emoji_dictionary)
	
		print("")
		end_beginning = datetime.datetime.now().time()
		print("Start time: ")
		print(start_beginning)
		print("End time: ")
		print(end_beginning)
	
	return
#-----------------------------------------------------------------------------------------------#

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
	from learn_constructions import learn_constructions

	try:
		pm = importlib.import_module(parameters_file)
	except ImportError:
		print("Error in specified parameters file. Format is 'files_parameters.FILENAME'")

	learn_constructions(pm.input_folder, 
						pm.training_files,
						pm.training_flag,
						pm.output_folder, 
						pm.output_files, 
						pm.max_construction_length,
						pm.nickname,
						pm.frequency_threshold_constructions,
						pm.encoding_type,
						pm.annotation_types, 
						pm.number_of_cpus_pruning,
						pm.number_of_cpus_merging,
						pm.coverage_threshold,
						pm.debug_file
						)
						
	#END CODE FOR RUNNING FROM COMMAND LINE#