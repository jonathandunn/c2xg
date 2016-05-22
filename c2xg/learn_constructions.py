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
							output_folder, 
							output_files, 
							frequency_threshold_constructions, 
							max_construction_length, 
							pairwise_threshold_lr, 
							pairwise_threshold_rl,
							nickname,
							encoding_type = "utf-8", 
							annotation_types = ["Lem", "Pos", "Cat"], 
							number_of_cpus_pruning = 1,
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

		from functions_candidate_pruning.prune_association import prune_association
		from functions_candidate_pruning.prune_horizontal import prune_horizontal
		from functions_candidate_pruning.prune_vertical import prune_vertical
		from functions_candidate_pruning.rank_constructions import rank_constructions
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
	
		input_dictionary = merge_output(output_files, frequency_threshold_constructions)
	
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
		temp_list = get_formatted_candidates(candidate_dictionary, frequency_threshold_constructions)
		candidate_list_formatted = temp_list[0]
		candidate_list_all = temp_list[1]
		candidate_list_pairs = temp_list[2]
		del temp_list
		
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
		feature_vector_list = pool_instance.map(partial(process_pairwise_feature_vector, 
															pairwise_dictionary=pairwise_dictionary
															), candidate_list_all, chunksize = 10000)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for candidate evaluation#
	
		vector_dataframe = pd.DataFrame(feature_vector_list, columns=['Candidate', 
														'Frequency', 
														'Summed_LR',
														'Smallest_LR',
														'Summed_RL', 
														'Smallest_RL',
														'Normalized_Summed_LR', 
														'Normalized_Summed_RL', 
														'Beginning_Reduced_LR',
														'Beginning_Reduced_RL',
														'End_Reduced_LR',
														'End_Reduced_RL',
														'Directional_Scalar',
														'Directional_Categorical'
														])
														
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
		unitwise_vector_list = pool_instance.map(partial(process_unitwise_feature_vector, 
															candidate_frequency_dict=candidate_frequency_dict, 
															lemma_frequency=lemma_frequency, 
															lemma_list=lemma_list, 
															pos_frequency=pos_frequency, 
															pos_list=pos_list, 
															category_frequency=category_frequency, 
															category_list=category_list, 
															total_units=total_units
															), candidate_list_all, chunksize = 50000)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for candidate evaluation#
		
		unitwise_vector_dataframe = pd.DataFrame(unitwise_vector_list, columns=['Candidate', 
																				'Beginning_Divided_LR', 
																				'Beginning_Divided_RL', 
																				'End_Divided_LR', 
																				'End_Divided_RL'
																			])
														
		full_vector_dataframe = pd.merge(vector_dataframe, unitwise_vector_dataframe, on='Candidate')
		
		output_file = output_folder + "/" + nickname + ".OutputVectors"
		output_file_pruned = output_folder + "/" + nickname + ".OutputVectorsPruned"
				
		write_results(full_vector_dataframe, 
						lemma_list, pos_list, 
						category_list, 
						output_file,
						encoding_type = encoding_type, 
						)
		
		write_candidates(data_file_vectors, full_vector_dataframe)
		
		end_time = time.time()
		print("\tCreate unitwise vectors: " + str(end_time - start_time))
		

		#---------------------------------------------------------------------------------------------#
		#3: Prune candidate constructions ------------------------------------------------------------#
		#---------------------------------------------------------------------------------------------#
		print("")
		print("Begin feature pruning.")
		print("")
	
		pruned_vector_dataframe = prune_association(full_vector_dataframe, pairwise_threshold_lr, pairwise_threshold_rl)
		pruned_vector_dataframe = prune_horizontal(pruned_vector_dataframe)
		pruned_vector_dataframe = prune_vertical(pruned_vector_dataframe)
		pruned_vector_dataframe = rank_constructions(pruned_vector_dataframe)
	
		write_results_pruned(pruned_vector_dataframe, 
								lemma_list, 
								pos_list, 
								category_list, 
								output_file_pruned, 
								encoding_type
								)
	
		#Create .model file for autonomous feature extraction#
		write_model(lemma_list, 
						pos_list, 
						word_list, 
						category_list, 
						semantic_category_dictionary, 
						sequence_list, 
						max_construction_length, 
						annotation_types, 
						pruned_vector_dataframe, 
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
						pm.output_folder, 
						pm.output_files, 
						pm.frequency_threshold_constructions, 
						pm.max_construction_length,
						pm.pairwise_threshold_lr, 
						pm.pairwise_threshold_rl,
						pm.nickname,
						pm.encoding_type,
						pm.annotation_types, 
						pm.number_of_cpus_pruning						
						)
						
	#END CODE FOR RUNNING FROM COMMAND LINE#