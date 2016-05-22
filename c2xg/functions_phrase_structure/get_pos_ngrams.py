#---------------------------------------------------------------------------------------------#
#INPUT: DataFrame and n-gram parameters ------------------------------------------------------#
#OUTPUT: List of ngrams, as tuples of index values -------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_pos_ngrams(pos_index, 
					stop, 
					current_df, 
					direction_flag
					):

	import pandas as pd
	import cytoolz as ct
	
	from functions_phrase_structure.get_search_df import get_search_df
	from functions_phrase_structure.get_query_string import get_query_string
	from functions_phrase_structure.get_column_names import get_column_names
	from functions_phrase_structure.get_ngram_count import get_ngram_count
	
	full_ngram_list = []
	
	for i in range(2, stop+1):
	
		#Create search DataFrame and keep only sequences within the same sentence starting with current POS#
		ngram_df = get_search_df(current_df, i)
		query_string = get_query_string(i, pos_index, direction_flag)
		ngram_df = ngram_df.query(query_string, parser='pandas', engine='numexpr')
		
		#Remove sentence columns#
		column_names = get_column_names(i)
		ngram_df = ngram_df.loc[:,column_names]
		
		#Sort results#
		ngram_df = ngram_df.fillna(value=0)
		column_list = ngram_df.columns.values.tolist()
		ngram_df = ngram_df[column_list].astype(int)			
		ngram_df = ngram_df.sort_values(by=column_list, axis=0, ascending=True, inplace=False, kind="mergesort")
		
		current_ngram_dictionary = get_ngram_count(ngram_df)
			
		full_ngram_list.append(current_ngram_dictionary)
			
		del ngram_df
		del current_ngram_dictionary
				
	return full_ngram_list
#---------------------------------------------------------------------------------------------#