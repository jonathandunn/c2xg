#---------------------------------------------------------------------------------------------#
#INPUT: Data files, current pos tag, and ngram parameters ------------------------------------#
#OUTPUT: Pos tag and head status: Non-Head, Head-First, Head-Last ----------------------------#
#Take data files and pos tags and return head status for each tag ----------------------------#
#---------------------------------------------------------------------------------------------#
def get_matrix(pos_list, 
				data_files, 
				distance_threshold,
				number_of_cpus,
				encoding_type,
				semantic_category_dictionary,
				word_list,
				lemma_list,
				lemma_dictionary,
				pos_dictionary,
				category_dictionary,
				delete_temp,
				run_parameter = 0
				):

	#--For multi-processing--#
	if run_parameter == 0:
		run_parameter = 1
	#-----------------------#
		
		import pandas as pd
		import cytoolz as ct
		import multiprocessing as mp
		from functools import partial
		import time
		import itertools
		
		from functions_input.pandas_open import pandas_open
		from functions_phrase_structure.update_base_frequencies import update_base_frequencies
		from functions_phrase_structure.update_pair_frequencies import update_pair_frequencies
		from functions_phrase_structure.get_association import get_association
		from functions_phrase_structure.check_distance import check_distance
		
		time_start = time.time()
		
		lr_association_dictionary = {}
		rl_association_dictionary = {}
		file_counter = 0

		pair_frequency_dictionary = {}
		base_frequency_dictionary = {}
		total_units = 0
		
		#Loop through files to support out-of-memory datasets#
		while True:
			
			if file_counter >= len(data_files):
				print("Insufficient data: association values do not stabilize.")
				break
			
			file = data_files[file_counter]
			file_counter += 1
			
			print("")
			print("")
						
			current_df = pandas_open(file, 
										encoding_type,
										semantic_category_dictionary,
										word_list,
										lemma_list,
										pos_list,
										lemma_dictionary,
										pos_dictionary,
										category_dictionary,
										save_words = False,
										write_output = False,
										delete_temp = delete_temp
										)
			print("")
			
			#Update frequency index for individual units#
			base_frequency_dictionary = update_base_frequencies(base_frequency_dictionary, current_df)
			total_units += len(current_df)
			
			#Multi-process frequency index for pairs (PoS1, PoS2) by PoS1#
			pool_instance=mp.Pool(processes = number_of_cpus, maxtasksperchild = None)
			current_pair_frequency_dictionary = pool_instance.map(partial(update_pair_frequencies,
																			index_list = pos_list,
																			current_df = current_df,
																			base_frequency_dictionary = base_frequency_dictionary,
																			lemma_list = lemma_list
																			), [i for i in range(len(pos_list))], chunksize = 50)
			pool_instance.close()
			pool_instance.join()
	
			#Merge after multi-processing#
			current_pair_frequency_dictionary = ct.merge([x for x in current_pair_frequency_dictionary if isinstance(x, dict)])

			#If necessary, initialize for first run#
			if file_counter == 1:
				pair_frequency_dictionary = current_pair_frequency_dictionary
			
			#Update cumulative values#
			else:
			
				pair_frequency_dictionary = ct.merge_with(sum, pair_frequency_dictionary, current_pair_frequency_dictionary)
			
			#Multi-process for LR association#
			pool_instance=mp.Pool(processes = number_of_cpus, maxtasksperchild = None)
			current_lr_association_dictionary = pool_instance.map(partial(get_association,
																			direction = "LR", 
																			base_frequency_dictionary = base_frequency_dictionary,
																			pair_frequency_dictionary = pair_frequency_dictionary,
																			total_units = total_units,
																			pos_list = pos_list
																			), pair_frequency_dictionary.keys(), chunksize = 50)
			pool_instance.close()
			pool_instance.join()
			
			#Multi-process for RL association#
			pool_instance=mp.Pool(processes = number_of_cpus, maxtasksperchild = None)
			current_rl_association_dictionary = pool_instance.map(partial(get_association,
																			direction = "RL", 
																			base_frequency_dictionary = base_frequency_dictionary,
																			pair_frequency_dictionary = pair_frequency_dictionary,
																			total_units = total_units,
																			pos_list = pos_list
																			), pair_frequency_dictionary.keys(), chunksize = 50)
			pool_instance.close()
			pool_instance.join()
			
			#Merge after multi-processing#
			current_lr_association_dictionary = ct.merge([x for x in current_lr_association_dictionary if isinstance(x, dict)])												
			current_rl_association_dictionary = ct.merge([x for x in current_rl_association_dictionary if isinstance(x, dict)])
			
			#If first time through, set previous association dictionaries to current and stop#
			if file_counter == 1:
				previous_lr_association_dictionary = current_lr_association_dictionary
				previous_rl_association_dictionary = current_rl_association_dictionary
				print("\tInitializing previous values for first run.")
			
			#If not first time through, continue with association distance calculation#
			else:

				lr_distance = check_distance(current_lr_association_dictionary, 
													 previous_lr_association_dictionary
													 )
													 
				rl_distance = check_distance(current_rl_association_dictionary, 
													 previous_rl_association_dictionary
													 )
													 
				print("\tDistance between current and previous associations: " + str(lr_distance) + "(LR) and " + str(rl_distance) + "(RL).")
				
				#Set previous dictionaries for the next cycle#
				previous_lr_association_dictionary = current_lr_association_dictionary
				previous_rl_association_dictionary = current_rl_association_dictionary

				if lr_distance < distance_threshold and rl_distance < distance_threshold:
				
					print("Stable association distance threshold reached.")
					break
	
	print("Time for creating matrix: " + str(time.time() - time_start))
	
	#Ensure unobserved pairs also present in matrix#
	complete_pairs = []
	
	for i in range(len(pos_list)):
		for j in range(len(pos_list)):
		
			if pos_list[i] != "n\a" and pos_list[j] != "n\a":
				pair = (i, j)
				complete_pairs.append(pair)
				
	for pair in complete_pairs:
		if pair not in pair_frequency_dictionary:
			pair_frequency_dictionary[pair] = 0
			current_lr_association_dictionary[pair] = 0
			current_rl_association_dictionary[pair] = 0
	
	return pair_frequency_dictionary, current_lr_association_dictionary, current_rl_association_dictionary, base_frequency_dictionary, file_counter
#---------------------------------------------------------------------------------------------#