#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

#-- Helper function for producing association vectors from saved candidate files

def get_association(Parameters, Grammar, training_files, output_suffix, freq_threshold, output_flag, run_parameter = 0):

	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1
		
		print("")
		print("Starting C2xG.Get_Association")
		print("")
		
		import datetime
		import time
		import sys
		import multiprocessing as mp
		import pandas as pd
		from functools import partial
		import cytoolz as ct

		from candidate_extraction.read_candidates import read_candidates
		from candidate_extraction.write_candidates import write_candidates

		from association_measures.get_formatted_candidates import get_formatted_candidates
		from association_measures.get_phrase_count import get_phrase_count
		from association_measures.get_pairwise_df import get_pairwise_df
		from association_measures.get_dictionary import get_dictionary
		from association_measures.process_pairwise_feature_vector import process_pairwise_feature_vector
		from association_measures.process_unitwise_feature_vector import process_unitwise_feature_vector
		from association_measures.get_frequency_dict import get_frequency_dict
		from association_measures.write_results import write_results
		from association_measures.merge_output import merge_output
		from association_measures.get_df_pairwise import get_df_pairwise
		from association_measures.get_df_unitwise import get_df_unitwise
		
		from candidate_selection.write_results_pruned import write_results_pruned
		from candidate_selection.write_model import write_model
	
		start_beginning = datetime.datetime.now().time()
	
		#---------------------------------------------------------------------------------------------#
		#1: Merge provided output files and return necessary data ------------------------------------#
		#---------------------------------------------------------------------------------------------#
		output_files = [x.replace("/Temp/","/Temp/Candidates/") + output_suffix for x in training_files]
		input_dictionary, Grammar = merge_output(output_files, freq_threshold, Parameters.CPUs_Merging)
	
		sequence_list = input_dictionary['sequence_list']
		lemma_frequency = input_dictionary['lemma_frequency']
		pos_frequency = input_dictionary['pos_frequency']
		category_frequency = input_dictionary['category_frequency']
		number_of_words = input_dictionary['number_of_words']
		candidate_dictionary = input_dictionary['candidate_dictionary']
		
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
		
		if len(candidate_list_formatted) < 10:
			print("WARNING: Fewer than 10 candidates discovered.")
			
		long_check = [x for x in candidate_list_formatted if len(x) > 2]
		long_check = len(long_check)
		
		del temp_list
		del candidate_dictionary

		print("Creating dictionary of A, B, C, D co-occurrence statistics.")
		start_time = time.time()
		#Start multi-processing for creating pairwise A, B, C, D dataframe#
		pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = None)
		pairwise_list = pool_instance.map(partial(get_pairwise_df, 
														lemma_frequency=lemma_frequency, 
														lemma_list=Grammar.Lemma_List, 
														pos_frequency=pos_frequency, 
														pos_list=Grammar.POS_List, 
														category_frequency=category_frequency, 
														category_list=Grammar.Category_List, 
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
		pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = None)
		unweighted_feature_vector_list = pool_instance.map(partial(process_pairwise_feature_vector, 
															pairwise_dictionary=pairwise_dictionary,
															freq_weighted=False
															), candidate_list_all, chunksize = 10000)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for candidate evaluation#
		
		#Start multi-processing for candidate evaluation#
		pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = None)
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
		pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = None)
		unweighted_unitwise_vector_list = pool_instance.map(partial(process_unitwise_feature_vector, 
															candidate_frequency_dict=candidate_frequency_dict, 
															lemma_frequency=lemma_frequency, 
															lemma_list=Grammar.Lemma_List, 
															pos_frequency=pos_frequency, 
															pos_list=Grammar.POS_List, 
															category_frequency=category_frequency, 
															category_list=Grammar.Category_List, 
															total_units=total_units,
															freq_weighted=False
															), candidate_list_all, chunksize = 50000)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing for candidate evaluation#
		
		#Start multi-processing for candidate evaluation#
		pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = None)
		weighted_unitwise_vector_list = pool_instance.map(partial(process_unitwise_feature_vector, 
															candidate_frequency_dict=candidate_frequency_dict, 
															lemma_frequency=lemma_frequency, 
															lemma_list=Grammar.Lemma_List, 
															pos_frequency=pos_frequency, 
															pos_list=Grammar.POS_List, 
															category_frequency=category_frequency, 
															category_list=Grammar.Category_List, 
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
		
		#Clean memory#
		del pairwise_vector_df
		del unitwise_vector_df
		del candidate_list_all
		del candidate_list_pairs
		del pairwise_dictionary
		del candidate_frequency_dict
		
		end_time = time.time()
		print("\tCreate unitwise vectors: " + str(end_time - start_time))
	
		if output_flag == "Write":
		
			output_file = Parameters.Output_Folder + "/" + Parameters.Nickname + ".OutputVectors"
			output_file_pruned = Parameters.Output_Folder + "/" + Parameters.Nickname + ".OutputVectorsPruned"
				
			write_results(full_vector_dataframe, 
							Grammar.Lemma_List, 
							Grammar.POS_List, 
							Grammar.Category_List, 
							output_file,
							encoding_type = Parameters.Encoding_Type 
							)
		
			write_candidates(data_file_vectors, full_vector_dataframe)
			
			return		

		elif output_flag == "Pass":
		
			return full_vector_dataframe, candidate_list_formatted
#-----------------------------------------------------------------------------------------------#