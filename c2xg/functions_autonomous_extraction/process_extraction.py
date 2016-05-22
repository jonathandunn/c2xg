#---------------------------------------------------------------------------------------------#
#FUNCTION: process_extraction --------------------------------------------------------#
#INPUT: Expanded DataFrame and list of constructions  ----------------------------------------#
#OUTPUT: Count in DataFrame for each construction, sentences covered, and examples -----------#
#---------------------------------------------------------------------------------------------#
def process_extraction(candidate_list, 
								max_construction_length, 
								input_dataframe, 
								lemma_list, 
								pos_list, 
								category_list, 
								number_of_sentences,
								full_scope = True,
								relative_freq = False,
								use_centroid = False,
								write_examples = ""
								):
	
	import pandas as pd
	import cytoolz as ct
	from sklearn.preprocessing import binarize
	from functions_autonomous_extraction.find_constructions import find_constructions
	from functions_autonomous_extraction.find_units import find_units
	from functions_autonomous_extraction.get_sentence_length_column import get_sentence_length_column
	from functions_autonomous_extraction.get_relative_frequencies import get_relative_frequencies
	
	total_sentences = []
	feature_dictionary = {}
	
	#First, evaluate string candidates to lists and sort by length#
	eval_list = []
	
	for construction in candidate_list:
		eval_list.append(eval(construction))

	eval_list = ct.groupby(len, eval_list)
	
	#Second, call search function by length#
	vector_list = []
	
	if full_scope == True:
	
		for i in eval_list.keys():
			
			if eval_list[i]:
					
				current_df = input_dataframe.copy(deep=True)
					
				print("")
				print("Starting constructions of length " + str(i) + ": " + str(len(eval_list[i])))
					
				#Returns list of [dictionary of candidate counts, total sentences represented]
				current_results_df = find_constructions(i, 
														eval_list[i], 
														current_df, 
														lemma_list, 
														pos_list, 
														category_list, 
														number_of_sentences,
														write_examples
														)
				vector_list.append(current_results_df)

				del current_results_df
				del current_df
			
			#POS Features#			
			current_df = input_dataframe.copy(deep=True)
			print("")
			print("Starting POS units: " + str(len(pos_list)))
			current_results_df = find_units(current_df, 
											pos_list, 
											'Pos', 
											number_of_sentences, 
											lemma_list, 
											pos_list, 
											category_list
											)
											
			vector_list.append(current_results_df)

			del current_results_df
			del current_df
			
			#Semantic Category Features#
			current_df = input_dataframe.copy(deep=True)
			print("")
			print("Starting Semantic units: " + str(len(category_list)))
			current_results_df = find_units(current_df, 
											category_list, 
											'Cat', 
											number_of_sentences, 
											lemma_list, 
											pos_list, 
											category_list
											)
											
			vector_list.append(current_results_df)
			print("")
			del current_results_df
			del current_df
		
	#Outside full feature check#
	#Now add lexical features#
	current_df = input_dataframe.copy(deep=True)
	print("")
	print("Starting Lemma units: " + str(len(lemma_list)))
	current_results_df = find_units(current_df, 
									lemma_list, 
									'Lem', 
									number_of_sentences, 
									lemma_list, 
									pos_list, 
									category_list
									)
											
	vector_list.append(current_results_df)

	del current_results_df
	del current_df

	if relative_freq == True:
		#Now get sentence_length column for calculating relative frequency#
		sentence_length_df = get_sentence_length_column(input_dataframe, number_of_sentences)
		vector_list.insert(0,sentence_length_df)
	
	#Now create DataFrame and name the columns#
	full_vector = pd.concat(vector_list, axis=1)
	
	#Convert frequencies to relative frequencies if requested#
	if relative_freq == True:
		full_vector = get_relative_frequencies(full_vector)
		
	#Binarize for usage probabilities#
	if use_centroid == True:
			
		column_list = full_vector.columns
		binary_array = binarize(full_vector, threshold = 0.9)
		full_vector = pd.DataFrame(binary_array, columns = column_list)
		
	return full_vector
#---------------------------------------------------------------------------------------------#