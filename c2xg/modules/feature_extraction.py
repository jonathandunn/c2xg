#General modules
import time
import codecs
import pandas as pd
import cytoolz as ct
import numpy as np
from sklearn.preprocessing import binarize

#C2xG modules
from modules.candidate_extraction import read_candidates
from modules.candidate_extraction import create_shifted_df
from modules.candidate_extraction import get_query

#INPUT: Expanded DataFrame and list of constructions  ----------------------------------------#
#OUTPUT: Count in DataFrame for each construction, sentences covered, and examples -----------#

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
#---------------------------------------------------------------------------------------------#

def print_constructs(search_df, 
						candidate, 
						lemma_list, 
						pos_list, 
						category_list, 
						write_examples
						):

	with codecs.open(write_examples, "a", encoding = "utf-8") as fo:
	
		candidate_string = ""
		candidate_flag = 0
		
		#Write candidate name to file#
		for tuple_pair in candidate:
			
			representation = tuple_pair[0]
			index_value = tuple_pair[1]
			
			if representation == "Lex":
				item_value = "'" + str(lemma_list[index_value]) + "'"
			
			elif representation == "Pos":
				item_value = pos_list[index_value].upper()
			
			elif representation == "Cat":
				item_value = "<" + str(category_list[index_value]) + ">"
				
			if candidate_flag > 0:
				candidate_string += " -- "
			
			candidate_string += item_value
			candidate_flag += 1
				
		fo.write(str(candidate_string))
		fo.write(str("\n"))

		#Finished writing candidate name#
		
		#Limit search_df to only lexical representation#
		column_list = search_df.columns
		new_columns = []
		
		for name in column_list:
			if name[0:3] == "Lex":
				new_columns.append(name)
		
		search_df = search_df.loc[:, new_columns]
		#Finished limiting search_df#
		
		for row in search_df.itertuples(index = False, name = "None"):

			for annotation in row:
			
				fo.write(str("\t"))
				fo.write(str(lemma_list[annotation]))
				fo.write(str(" "))
				
			fo.write(str("\n"))
	
	return
#---------------------------------------------------------------------------------------------#
#INPUT: List of (possibly duplicated) sentences marking each occurrence of current feature ---#
#OUTPUT: Series with relative frequency for current construction -----------------------------#

def get_vector_column(current_sentences, number_of_sentences):
	
	series_list = []
	
	frequency_index = ct.frequencies(current_sentences)
	
	for i in range(1, number_of_sentences+1):
	
		try:
			series_list.append(frequency_index[i])
			
		except:
			series_list.append(0)
	
	return  series_list
#---------------------------------------------------------------------------------------------#
#INPUT: Expanded DataFrame -------------------------------------------------------------------#
#OUTPUT: Dataframe with length of each sentence / text in words ------------------------------#

def get_sentence_length_column(current_df, number_of_sentences):
		
	start_all = time.time()
	length_list = []
	
	search_df = current_df.loc[:,['Sent', 'Alt', "Lex"]]
	search_df = search_df.query("(Alt == 0)", parser='pandas', engine='numexpr')

	current_sentences = search_df.loc[:,'Sent'].tolist()
	
	frequency_index = ct.frequencies(current_sentences)
	
	for i in range(1, number_of_sentences+1):

		try:
			length_list.append(frequency_index[i])
			
		except:
			length_list.append(0)
	
	temp_series = pd.Series(length_list, name = "Length")
	temp_series.index = np.arange(1, len(temp_series)+1)
	
	end_all = time.time()
	print("Total time for sentence length column: " + str(end_all - start_all))
	
	return temp_series
#---------------------------------------------------------------------------------------------#
#INPUT: Full feature DataFrame ---------------------------------------------------------------#
#OUTPUT: DataFrame with frequencies relative to total words ----------------------------------#

def get_relative_frequencies(full_vector):
	
	length_series = full_vector.loc[:,'Length'].copy('deep')
	
	full_vector = full_vector.div(full_vector.loc[:,'Length'], axis=0)
	
	full_vector.loc[:,'Length'] = length_series
	
	return  full_vector
#---------------------------------------------------------------------------------------------#
#INPUT: List of columns to check for equivalence ---------------------------------------------#
#OUTPUT: String of query ---------------------------------------------------------------------#

def get_query_autonomous_zero(column_names):
		
	query = ""
	column_names.remove('Sent')
	
	for i in range(len(column_names)):
		
		if i == 0:
			query = "(" + str(column_names[i]) + " != 0 "
				
		elif i  > 0:
			query += " and " + str(column_names[i]) + " != 0"
	
	query += ")"
	
	return query
#---------------------------------------------------------------------------------------------#
#INPUT: List of columns to check for equivalence ---------------------------------------------#
#OUTPUT: String of query ---------------------------------------------------------------------#

def get_query_autonomous_candidate(current_candidate):
		
	query = ""
	
	for i in range(len(current_candidate)):
	
		current_col = current_candidate[i][0]
		current_index = current_candidate[i][1]
		
		if i == 0:
			query = "(" + str(current_col) + str(i) + " == " + str(current_index) + " "
			
		elif i  > 0:
			query += "and " + str(current_col) + str(i) + " == " + str(current_index) + " "
	
	query += ")"
	
	return query
#---------------------------------------------------------------------------------------------#
#INPUT: Full Vector DataFrame, and MetaData as (ID, DICTIONARY) tuples -----------------------#
#OUTPUT: Dataframe with meta-data columns added per text -------------------------------------#

def get_meta_data(full_vector_df, metadata_tuples):
	
	#Get fields for saved meta-data#
	temp_dictionary = metadata_tuples[0][1]	
	metadata_columns = list(temp_dictionary.keys())
	
	#Initialize dictionary of lists for field values#
	column_dictionary = {}
	
	for column_name in metadata_columns:
		column_dictionary[column_name] = []
	
	#For each vector (e.g., text), get its meta-data, add to lists#
	for text_id in metadata_tuples:

		current_dictionary = text_id[1]
		
		for column_name in list(current_dictionary.keys()):
			column_dictionary[column_name].append(current_dictionary[column_name])
			
	#For each field, create series and add to vector DataFrame#
	for column_name in list(column_dictionary.keys()):

		temp_series = pd.Series(column_dictionary[column_name], name = column_name)
		temp_series.index = np.arange(1, len(temp_series) + 1)
		full_vector_df = pd.concat([full_vector_df, temp_series], axis = 1)

		del temp_series		
	
	return full_vector_df
#---------------------------------------------------------------------------------------------#
#INPUT: DataFrame with current construction matches and  lemma, pos, category index lists ----#
#OUTPUT: list readable examples to write -----------------------------------------------------#

def get_examples(search_df, 
					candidate, 
					lemma_list, 
					pos_list, 
					category_list, 
					current_length
					):
	
	candidate_id = ""
	example_list = []
	
	for i in range(len(candidate)):
	
		current_col = candidate[i][0]
		current_index = candidate[i][1]
		
		if current_col == "Lex":
			index_list = lemma_list
		
		elif current_col == "Pos":
			index_list = pos_list
		
		elif current_col == "Cat":
			index_list = category_list
			
		current_unit = index_list[current_index]
		
		candidate_id += str(current_col) + ":" + str(current_unit) + " "
		
	example_list.append(candidate_id)
	
	#Start loop through rows#
	column_list = [2]
	previous = 2
	
	for i in range(1,current_length):
		column_list.append(previous+3)
		previous = previous + 3
		
	for row in search_df.itertuples():
	
		current_row = ""
		
		for column in column_list:
			
			if column == 2:
				current_row += str(lemma_list[row[column]])
			
			else:
				current_row += " " + str(lemma_list[row[column]])
			
		example_list.append(current_row)
		
	return example_list
#---------------------------------------------------------------------------------------------#
#INPUT: Full Vector DataFrame ----------------------------------------------------------------#
#OUTPUT: Dataframe with column indicating how many non-sparse features each row has ----------#

def get_coverage_column(full_vector_df):
		
	count_series = (full_vector_df != 0).astype(int).sum(axis=1)
	full_vector_df.loc[:,'Coverage'] = count_series
	
	return full_vector_df
#---------------------------------------------------------------------------------------------#
#INPUT: Candidate construction and index lists -----------------------------------------------#
#OUTPUT: String of readable construction id --------------------------------------------------#

def get_construction_name(candidate, 
							lemma_list, 
							pos_list, 
							category_list
							):
	
	candidate_id = ""
	
	for i in range(len(candidate)):
	
		current_col = candidate[i][0]
		current_index = candidate[i][1]
		
		if current_col == "Lex":
			index_list = lemma_list
		
		elif current_col == "Pos":
			index_list = pos_list
		
		elif current_col == "Cat":
			index_list = category_list
			
		current_unit = index_list[current_index]
		
		candidate_id += str(current_col) + ":" + str(current_unit) + " "
		
	candidate_id = candidate_id.encode()
		
	return candidate_id
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def get_centroid_normalization(full_vector_df, centroid_df):
		
	#Number of occurrences = n (this is the raw frequency value in the input feature vector
	#Normalized value = Inv Probability added n times

	#Move into NumPy#
	full_array = full_vector_df.values
	centroid_array = centroid_df.values
	
	#Centroid probabilities to the power of feature occurrence per text#
	full_array = np.power(centroid_array, full_array)

	#Invert probabilities to give unexpected features more weight#
	length = full_array.shape[1]
	a = np.empty(length)
	a.fill(1)
	full_array = np.subtract(a, full_array)
	
	column_list = full_vector_df.columns
	
	return_df = pd.DataFrame(full_array)
	return_df.columns = column_list
	
	return_df.index = np.arange(1, len(return_df)+1)

	return return_df
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def get_centroid(vector_file_list, delete_temp):
	
	pd.set_option('precision',12)
	
	#The centroid is the expected usage of a given feature across the whole dataset.
	#This is calculated as the inverse probability of a text containing the feature.
	
	vector_file_list = vector_file_list[0]
	centroid_list = []
	
	#First merge all intermediate centroid files#
	print("Merging centroids from individual input files.")
	for vector_file in vector_file_list:
		
		temp_vector = read_candidates(vector_file)
		centroid_list.append(temp_vector)
	
	#Now concat and sum to get total texts each feature occurs in#
	full_vector = pd.concat(centroid_list, axis = 1)
	summed_vector = full_vector.sum(axis = 1)
	
	#Get total instances, drop instances and length as no longer necessary#
	total_instances = summed_vector.loc["Instances"]
	summed_vector.drop('Instances', axis = 0, inplace = True)
	print("Total instances represented: " + str(total_instances))
	
	#Get probability of each feature occur in text#
	summed_vector = summed_vector.div(total_instances, level = None, fill_value = 0, axis = 0)

	#Transpose to allow row by row manipulations during feature extraction#
	centroid_df = pd.DataFrame(summed_vector)
	centroid_df = centroid_df.T

	#---Possibly delete temp centroids---#
	if delete_temp == True:
		import os
		
		for file in vector_file_list:
			os.remove(file)
	#------------------------------------#
	
	return  centroid_df
#---------------------------------------------------------------------------------------------#
#INPUT: Current DataFrame and index lists ----------------------------------------------------#
#OUTPUT: DataFrame with matches to template --------------------------------------------------#

def find_units(current_df, 
				unit_list, 
				unit_type, 
				number_of_sentences, 
				lemma_list, 
				pos_list, 
				category_list
				):
    	
	start_all = time.time()	
	
	vector_column_list = []
	column_names = []
	
	#Make zero list for non-matches#
	zero_list = []
	for i in range(1,number_of_sentences+1):
		zero_list.append(0)
	
	search_df = current_df.loc[:,['Sent', 'Alt', str(unit_type)]]
	query_string = "(Alt == 0)"
	search_df = search_df.query(query_string, parser='pandas', engine='numexpr')
	
	for i in range(1,len(unit_list)):
	
		#Check to prevent lemma phrases from being counted#
		if unit_type == "Lex" and "_PHRASE" in unit_list[i]:
			print("", end="")
		
		#If not a lemma phrase, proceed as planned#
		else:
			candidate = [(unit_type, i)]
		
			query_string = "(" + unit_type + " == " + str(i) + ")"
			match_df = search_df.query(query_string, parser='pandas', engine='numexpr')
		
			#Check to prevent counting of non-existent features#
			if len(match_df) > 0:
		
				current_sentences = []
			
				#Create vector for current construction#
				current_sentences = match_df.loc[:,'Sent'].tolist()
				candidate_list = get_vector_column(current_sentences, number_of_sentences)
				candidate_id = get_construction_name(candidate, lemma_list, pos_list, category_list)
				temp_series = pd.Series(candidate_list, name = candidate_id)
				temp_series.index = np.arange(1, len(temp_series)+1)
				vector_column_list.append(temp_series)
				
				del temp_series
				del candidate_list
					
			#Process non-existent features#
			else:
			
				candidate_id = get_construction_name(candidate, lemma_list, pos_list, category_list)
				series_list = zero_list
				
				temp_series = pd.Series(series_list, name = candidate_id)
				temp_series.index = np.arange(1, len(temp_series)+1)
				vector_column_list.append(temp_series)

				del series_list
				del temp_series
			
			del match_df
			#Done with feature match check#
		#Done with lemma phrase check#	
		
	#Done counting features. Now create DataFrame for results#
	results_df = pd.concat(vector_column_list, axis = 1)
	
	end_all = time.time()
	print("Total time for extraction of " + str(unit_type) + ": " + str(end_all - start_all))
	
	return results_df
#---------------------------------------------------------------------------------------------#
#INPUT: Current template and DataFrame -------------------------------------------------------#
#OUTPUT: DataFrame with matches to template --------------------------------------------------#

def find_constructions(current_length, 
						candidate_list, 
						current_df, 
						lemma_list, 
						pos_list, 
						category_list, 
						number_of_sentences,
						write_examples = ""
						):
    
	vector_column_list = []
	start_all = time.time()

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
	
	#Make zero list for non-matches#
	zero_list = []
	for i in range(1,number_of_sentences+1):
		zero_list.append(0)
	
	#Now, search for individual sequences within prepared DataFrame#
	column_names = []
	
	for candidate in candidate_list:
	
		candidate_query = get_query_autonomous_candidate(candidate)
		search_df = current_df.query(candidate_query, parser='pandas', engine='numexpr')

		#Find duplicated rows within same sentence and remove those which are duplicated#
		column_list = search_df.columns.values.tolist()
		row_mask = search_df.duplicated(subset=column_list, keep="first")
		search_df = search_df.loc[~row_mask,]
		del row_mask
		
		#If no matches, just use empty series#
		if len(search_df) > 0:
			
			#If using multiple Alts, ensure no duplicate representations#
			if current_length > 1:
				search_df = search_df.drop_duplicates(subset = 'Mas', keep = "first")

			#Check if need to write examples of constructions#
			if write_examples != "":
				from feature_extraction.print_constructs import print_constructs
				print_constructs(search_df, candidate, lemma_list, pos_list, category_list, write_examples)
			#Done writing constructions if necessary#
			
			#Create vector for current construction#
			current_sentences = search_df.loc[:,'Sent'].tolist()
			candidate_list = get_vector_column(current_sentences, number_of_sentences)
			candidate_id = get_construction_name(candidate, lemma_list, pos_list, category_list)
			temp_series = pd.Series(candidate_list, name = candidate_id)
			temp_series.index = np.arange(1, len(temp_series)+1)
			vector_column_list.append(temp_series)
			
			del temp_series
			del candidate_list
					
		else:

			candidate_id = get_construction_name(candidate, lemma_list, pos_list, category_list)
			series_list = zero_list
			
			temp_series = pd.Series(series_list, name = candidate_id)
			temp_series.index = np.arange(1, len(temp_series)+1)
			vector_column_list.append(temp_series)

			del series_list
			del temp_series
						
	#Done counting features. Now create DataFrame for results#
	results_df = pd.concat(vector_column_list, axis = 1)

	end_all = time.time()
	print("Total time for extraction of constructions of length " + str(current_length) + ": " + str(end_all - start_all))
	
	return results_df
#---------------------------------------------------------------------------------------------#
#INPUT: template -----------------------------------------------------------------------------#
#OUTPUT: Take a dataframe, the column to repeat, and a listof times to repeat ----------------#
#Specific to creating alt / sent dataframes b/c more efficient than a generalized version ----#

def create_shifted_length_df(original_df, current_length):
	
	column_list = []
	
	ordered_columns = []
	named_columns = []
	
	ordered_columns.append(['Sent', 'Mas'])
	named_columns.append('Sent')
	named_columns.append('Mas')
	
	for i in range(current_length):
		ordered_columns.append(["Lex", 'Pos', 'Cat'])
		named_columns.append("Lex" + str(i))
		named_columns.append('Pos' + str(i))
		named_columns.append('Cat' + str(i))
		
	for i in range(len(ordered_columns)):
		holder_df = original_df.loc[:,ordered_columns[i]]
		column_list.append(holder_df.shift(-i))
		del holder_df
	
	original_df = pd.concat(column_list, axis=1)
	del column_list
	
	original_df.columns = named_columns
	
	return original_df
#---------------------------------------------------------------------------------------------#