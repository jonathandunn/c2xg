#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def prune_unobserved(data_files,
							active_list,
							direction,
							pos_list,
							encoding_type,
							semantic_category_dictionary,
							word_list,
							lemma_list,
							lemma_dictionary,
							pos_dictionary,
							category_dictionary,
							delete_temp
							):

	import pandas as pd
	import cytoolz as ct
	import time
		
	from functions_input.pandas_open import pandas_open
	from functions_constituent_reduction.find_unit_index import find_unit_index
	from functions_constituent_reduction.get_search_df_expansion import get_search_df_expansion
	from functions_constituent_reduction.get_expansion_query import get_expansion_query
	from functions_constituent_reduction.constituents_reduce import constituents_reduce
		
	time_start = time.time()
	
	constituent_len_dictionary = ct.groupby(len, active_list)
	length_list = list(constituent_len_dictionary.keys())
	length_list = sorted(length_list, reverse=False)
	
	count_dictionary = {}
	
	for data_file in data_files:
		
		#Loop through files to support out-of-memory datasets#
		current_df = pandas_open(data_file, 
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
		
		print("\tFinding constituent matches.")	
		#Loop through constituents by length, creating only 1 search_df for each length#
		for length in length_list:
		
			print("Starting length: " + str(length) + " with " + str(len(constituent_len_dictionary[length])) + " sequences.")
			
			#Generate initial search DF#
			copy_df = current_df.copy("Deep")
			search_df = get_search_df_expansion(copy_df, length)
			
			#Loop through constituents of current length#
			for constituent in constituent_len_dictionary[length]:
			
				if direction == "L":
					current_head = constituent[0]
				
				elif direction == "R":
					current_head = constituent[-1]

				#Find constituents#
				query_string = get_expansion_query(constituent)
				match_df = search_df.query(query_string, parser='pandas', engine='numexpr')
				
				current_count = len(match_df)
				
				count_dictionary[constituent] = current_count
				
			#Done looping through constituents of current length#
		#Done looping through constituents by length#
		
	above_zero = lambda x: x > 0
	frequency_dictionary = ct.valfilter(above_zero, count_dictionary)
	
	active_list = [x for x in active_list if x in frequency_dictionary.keys()]
	
	return active_list
#---------------------------------------------------------------------------------------------#