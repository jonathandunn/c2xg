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
								frequency_type,
								vector_type,
								write_examples = ""
								):
	
	import pandas as pd
	import cytoolz as ct
	from sklearn.preprocessing import binarize
	from feature_extraction.find_constructions import find_constructions
	from feature_extraction.find_units import find_units
	from feature_extraction.get_sentence_length_column import get_sentence_length_column
	from feature_extraction.get_relative_frequencies import get_relative_frequencies
	
	total_sentences = []
	feature_dictionary = {}
	vector_list = []
	
	#Settings
	if vector_type == "Lexical":
		extract_constructions = False
		extract_pos = False
		extract_cat = False
		extract_lex = True
	
	elif vector_type == "Units":
		extract_constructions = False
		extract_pos = True
		extract_cat = True
		extract_lex = True
	
	elif vector_type == "CxG":
		extract_constructions = True
		extract_pos = False
		extract_cat = False
		extract_lex = False
	
	elif vector_type == "CxG+Units":
		extract_constructions = True
		extract_pos = True
		extract_cat = True
		extract_lex = True
		
	if frequency_type == "Raw":
		relative_freq = False
		use_centroid = False
	
	elif frequency_type == "Relative":
		relative_freq = True
		use_centroid = False

	elif frequency_type == "TFIDF":
		relative_freq = False
		use_centroid = True
	
	#Get constructions
	if extract_constructions == True:
		eval_list = candidate_list
		
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
			
	#POS Features
	if extract_pos == True:
			
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
		
	#Semantic Category Features
	if extract_cat == True:
	
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
	
	#Lexical item features
	if extract_lex == True:

		current_df = input_dataframe.copy(deep=True)
		print("")
		print("Starting Lemma units: " + str(len(lemma_list)))
		current_results_df = find_units(current_df, 
										lemma_list, 
										"Lex", 
										number_of_sentences, 
										lemma_list, 
										pos_list, 
										category_list
										)
												
		vector_list.append(current_results_df)

		del current_results_df
		del current_df

	#Adjust raw frequencies as needed
	
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