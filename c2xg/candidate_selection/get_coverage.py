#---------------------------------------------------------------------------------------------#
#INPUT: Current template and DataFrame -------------------------------------------------------#
#OUTPUT: DataFrame with matches to template --------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_coverage(candidate, 
				current_df, 
				lemma_list, 
				pos_list, 
				category_list,
				total_words
				):
    
	import pandas as pd
	import cytoolz as ct
	import numpy as np

	from feature_extraction.get_query_autonomous_candidate import get_query_autonomous_candidate

	current_length = len(candidate)
	coverage_dictionary = {}
	coverage_dictionary[tuple(candidate)] = {}
	
	candidate_query = get_query_autonomous_candidate(candidate)
	search_df = current_df.query(candidate_query, parser='pandas', engine='numexpr')

	#Find duplicated rows within same sentence and remove those which are duplicated#
	column_list = search_df.columns.values.tolist()
	row_mask = search_df.duplicated(subset=column_list, keep="first")
	search_df = search_df.loc[~row_mask,]
	del row_mask
	
	list_of_indexes = []
	times_encoded = 0
	
	#If no matches, just use empty series#
	if len(search_df) > 0:
			
		search_df = search_df.drop_duplicates(subset = 'Mas', keep = "first")
		search_df = search_df.loc[:,["Mas", "EndMas"]]
		
		for row in search_df.itertuples(index = False, name = None):

			times_encoded += 1
			list_of_indexes += [x for x in range(row[0], row[1] + 1)]
			
	else:
	
		times_encoded = 0
		list_of_indexes = ()
		
	candidate_key = tuple(candidate)
	list_of_indexes = [tuple(list_of_indexes)]

	coverage_dictionary[candidate_key]["Encoded"] = times_encoded
	coverage_dictionary[candidate_key]["Indexes"] = list_of_indexes
	
	return coverage_dictionary
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#