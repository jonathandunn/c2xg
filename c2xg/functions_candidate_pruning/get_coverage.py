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
	
	from functions_autonomous_extraction.get_query_autonomous_candidate import get_query_autonomous_candidate

	current_length = len(candidate)
	coverage_dictionary = {}
	
	candidate_query = get_query_autonomous_candidate(candidate)
	search_df = current_df.query(candidate_query, parser='pandas', engine='numexpr')

	#Find duplicated rows within same sentence and remove those which are duplicated#
	column_list = search_df.columns.values.tolist()
	row_mask = search_df.duplicated(subset=column_list, keep="first")
	search_df = search_df.loc[~row_mask,]
	del row_mask
	
	covered = 0
	coverage = 0
	
	#If no matches, just use empty series#
	if len(search_df) > 0:
			
		search_df = search_df.drop_duplicates(subset = 'Mas', keep = "first")
		search_df = search_df.loc[:,["Mas", "EndMas"]]
		
		for row in search_df.itertuples(index = False, name = None):

			covered += (row[1] - row[0])
		
		if covered > 1:
			coverage = float(covered) / total_words
		else:
			coverage = 0

	if coverage < 0:
		coverage = 0
	
	candidate = tuple(candidate)
	coverage_dictionary[candidate] = coverage
						
	return coverage_dictionary
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#