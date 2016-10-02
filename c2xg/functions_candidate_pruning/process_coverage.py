#---------------------------------------------------------------------------------------------#
#Get indexes covered by each construction ----------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def process_coverage(training_files, 
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
					number_of_cpus,
					run_parameter = 0
					):
	
	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1
	
		import pandas as pd
		import cytoolz as ct
		import multiprocessing as mp
		from functools import partial
		
		from functions_candidate_pruning.get_coverage import get_coverage
		from functions_input.pandas_open import pandas_open
		from functions_constituent_reduction.expand_sentences import expand_sentences
		
		from functions_candidate_extraction.create_shifted_df import create_shifted_df
		from functions_candidate_extraction.get_query import get_query
		from functions_candidate_pruning.create_shifted_length_df import create_shifted_length_df
		from functions_autonomous_extraction.get_query_autonomous_zero import get_query_autonomous_zero
		
		from functions_candidate_extraction.read_candidates import read_candidates
		from functions_candidate_extraction.write_candidates import write_candidates
		
		#First, evaluate string candidates to lists and sort by length#
		eval_list = ct.groupby(len, candidate_list_formatted)
		first_flag = 1
		
		#Loop through all training files#
		for input_file in training_files:
		
			#If necessary, ingest and expand training files#
			if training_flag == "CONLL":
			
				#First, load and expand input file#
				input_df = pandas_open(input_file, 
											encoding_type, 
											semantic_category_dictionary, 
											word_list, 
											lemma_list, 
											pos_list, 
											lemma_dictionary, 
											pos_dictionary, 
											category_dictionary,
											write_output = False,
											delete_temp = False
											)
											
				total_words = len(input_df)
											
				input_df = expand_sentences(input_df, 
												lemma_list, 
												pos_list, 
												category_list, 
												encoding_type, 
												write_output = False, 
												phrase_constituent_list = phrase_constituent_list
												)
			
			#If training files are pre-saved, load them and the total word count#
			elif training_flag == "DF":
				
				temp_list = read_candidates(files)
				input_df = temp_list[0]
				total_words = temp_list[1]
				
				del temp_list
											
			#Second, call search function by length#
			result_list = []
			
			for i in eval_list.keys():
				
				current_length = i
				current_list = eval_list[i]
				
				if current_list:
							
					current_df = input_df.copy(deep=True)
							
					print("")
					print("Starting constructions of length " + str(i) + ": " + str(len(current_list)))
							
					if current_length > 1:
						
						#Create shifted alt-only dataframe for length of template#
						alt_columns = []
						alt_columns_names = []
						for i in range(current_length):
							alt_columns.append(1)
							alt_columns_names.append("c" + str(i))
						
						alt_dataframe = create_shifted_df(current_df, 1, alt_columns)
						alt_dataframe.columns = alt_columns_names
							
						query_string = get_query(alt_columns_names)
						row_mask_alt = alt_dataframe.eval(query_string)
						del alt_dataframe
					
						#Create shifted sent-only dataframe for length of template#
						sent_columns = []
						sent_columns_names = []
						for i in range(current_length):
							sent_columns.append(0)
							sent_columns_names.append("c" + str(i))
						
						sent_dataframe = create_shifted_df(current_df, 0, sent_columns)
						sent_dataframe.columns = sent_columns_names
						query_string = get_query(sent_columns_names)
						row_mask_sent = sent_dataframe.eval(query_string)
						del sent_dataframe
							
						#Create and shift template-specific dataframe#
						current_df = create_shifted_length_df(current_df, current_length)

						current_df = current_df.loc[row_mask_sent & row_mask_alt,]
						del row_mask_sent
						del row_mask_alt
						
						#Remove NaNS and change dtypes#
						current_df.fillna(value=0, inplace=True)
						column_list = current_df.columns.values.tolist()
						current_df = current_df[column_list].astype(int)
						
					elif current_length == 1:
						
						query_string = "(Alt == 0)"
						current_df = current_df.query(query_string, parser='pandas', engine='numexpr')
						current_df = current_df.loc[:,['Sent', 'Lem', 'Pos', 'Cat']]
						current_df.columns = ['Sent', 'Lem0', 'Pos0', 'Cat0']
					
					#Remove zero valued indexes#
					column_list = current_df.columns.values.tolist()
					query_string = get_query_autonomous_zero(column_list)
					current_df = current_df.query(query_string, parser='pandas', engine='numexpr')
					
					#Now, search for individual sequences within prepared DataFrame#
					
					#Start multi-processing#
					pool_instance=mp.Pool(processes = number_of_cpus, maxtasksperchild = None)
					coverage_list = pool_instance.map(partial(get_coverage, 
																current_df = current_df, 
																lemma_list = lemma_list, 
																pos_list = pos_list, 
																category_list = category_list,
																total_words = total_words
																), [x for x in current_list], chunksize = 500)
					pool_instance.close()
					pool_instance.join()
					#End multi-processing#

					coverage_list = ct.merge([x for x in coverage_list])
					result_list.append(coverage_list)
					
					del current_df
			
			#Merge and save coverage dictionaries for each training set#
			result_dict = ct.merge([x for x in result_list])
			result_df = pd.DataFrame.from_dict(result_dict, orient = "index")
			result_df.columns = [input_file]

			if first_flag == 1:
			
				return_df = result_df
				first_flag = 0
				
			else:

				return_df = pd.merge(return_df, result_df, left_index = True, right_index = True)
			
			del result_dict
			del result_df
		
		return return_df
#---------------------------------------------------------------------------------------------#