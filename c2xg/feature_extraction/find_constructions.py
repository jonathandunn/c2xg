#---------------------------------------------------------------------------------------------#
#INPUT: Current template and DataFrame -------------------------------------------------------#
#OUTPUT: DataFrame with matches to template --------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def find_constructions(current_length, 
						candidate_list, 
						current_df, 
						lemma_list, 
						pos_list, 
						category_list, 
						number_of_sentences,
						write_examples = ""
						):
    
	import pandas as pd
	import cytoolz as ct
	import numpy as np
	import time
	
	from candidate_extraction.create_shifted_df import create_shifted_df
	from candidate_extraction.get_query import get_query
	
	from feature_extraction.create_shifted_length_df import create_shifted_length_df
	from feature_extraction.get_query_autonomous_zero import get_query_autonomous_zero
	from feature_extraction.get_query_autonomous_candidate import get_query_autonomous_candidate
	from feature_extraction.get_vector_column import get_vector_column
	from feature_extraction.get_construction_name import get_construction_name
	
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
#---------------------------------------------------------------------------------------------#