#Get indexes covered by each construction ----------------------------------------------------#

def process_coverage(Parameters, 
						Grammar, 
						training_testing_files,
						testing_files,
						max_candidate_length, 
						candidate_list_formatted, 
						association_df, 
						expand_flag, 
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
		import math
		
		from candidate_selection.get_coverage import get_coverage
		from process_input.pandas_open import pandas_open
		from constituent_reduction.expand_sentences import expand_sentences
		from process_input.annotate_files import annotate_files
		from candidate_extraction.create_shifted_df import create_shifted_df
		from candidate_extraction.get_query import get_query
		from candidate_selection.create_shifted_length_df import create_shifted_length_df
		from feature_extraction.get_query_autonomous_zero import get_query_autonomous_zero
		from process_input.merge_conll import merge_conll	
		from process_input.merge_conll_names import merge_conll_names
		from candidate_extraction.read_candidates import read_candidates
		from candidate_extraction.write_candidates import write_candidates
		
		#-----Combine training_testing_files into one file for each restart and combine and testing_files into one file --#
		#-------This makes testing and restarts easier to handle ---------------------------------------------------------#
		
		print("\t\tJoining training-testing and testing files into single file for each restart with a single file for testing.")
		tuple_list, training_list, testing_list = merge_conll_names(training_testing_files, testing_files, Parameters)	

		#Start multi-processing#
		pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = None)
		pool_instance.map(partial(merge_conll, 
									encoding_type = Parameters.Encoding_Type
									), tuple_list, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing#

		#----FINISHED CONSOLIDATING TESTING SETS -----------------------------------------------------------------#
		
		#First, evaluate string candidates to lists and sort by length#
		eval_list = ct.groupby(len, candidate_list_formatted)
		for_list = training_list + testing_list
		
		for input_file in for_list:
		
			print("")
			print("\t\tBeginning MDL-prep for " + str(input_file))
			print("")
			
			input_file_original = input_file
			
			#First, load and expand input file#
			input_df = pandas_open(input_file, 
									Parameters, 
									Grammar,
									write_output = False,
									delete_temp = False
									)
												
			total_words = len(input_df)
												
			if expand_flag == True:
				input_df = expand_sentences(input_df, Grammar, write_output = False)
					
			else:
				input_df.loc[:,"Alt"] = 0
				input_df = input_df.loc[:,['Sent', 'Alt', 'Mas', "Lex", 'Pos', 'Cat']]

			#Save DF for later calculations#
			write_candidates(input_file + ".Training", input_df)
			
			#Second, call search function by length#
			result_list = []

			if input_df.empty:
				print("ERROR: Training DataFrame is empty.")
				sys.kill()
	
			for i in eval_list.keys():
					
				current_length = i
				current_list = eval_list[i]
	
				if current_list:
								
					current_df = input_df
								
					print("")
					print("\t\t\tStarting constructions of length " + str(i) + ": " + str(len(current_list)))
								
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
						current_df = current_df.loc[:,['Sent', "Lex", 'Pos', 'Cat']]
						current_df.columns = ['Sent', 'Lem0', 'Pos0', 'Cat0']
						
					#Remove zero valued indexes#
					column_list = current_df.columns.values.tolist()
					query_string = get_query_autonomous_zero(column_list)
					current_df = current_df.query(query_string, parser='pandas', engine='numexpr')

					#Now, search for individual sequences within prepared DataFrame#
					#Start multi-processing#
					pool_instance=mp.Pool(processes = Parameters.CPUs_General, maxtasksperchild = None)
					coverage_list = pool_instance.map(partial(get_coverage, 
																current_df = current_df, 
																lemma_list = Grammar.Lemma_List, 
																pos_list = Grammar.POS_List, 
																category_list = Grammar.Category_List,
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

			del result_list

			result_dict_encoded = {}
			result_dict_indexes = {}

			if len(result_dict.keys()) == 0:
			
				print("ERROR: No candidate coverage results for this datset.")
			
			else:
				for key in result_dict:

					result_dict_encoded[key] = result_dict[key]["Encoded"]
					result_dict_indexes[key] = result_dict[key]["Indexes"]
				
				del result_dict

				result_df_encoded = pd.DataFrame.from_dict(result_dict_encoded, orient = "index")
				result_df_encoded.columns = ["Encoded"]
				
				result_df_indexes = pd.DataFrame.from_dict(result_dict_indexes, orient = "index", dtype = "object")
				result_df_indexes.columns = ["Indexes"]
				
				result_df = pd.merge(result_df_encoded, result_df_indexes, left_index = True, right_index = True)
				index_list = [str(list(x)) for x in result_df.index.tolist()]
				result_df.loc[:,"Candidate"] = index_list

				del result_df_encoded
				del result_df_indexes
				
				result_df = pd.merge(result_df, association_df, on = "Candidate")

				write_candidates(input_file + ".MDL", result_df)
				
		training_list = [x for x in training_list]
		testing_list = [x for x in testing_list]
		
		return training_list, testing_list
#---------------------------------------------------------------------------------------------#