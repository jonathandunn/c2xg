#---------------------------------------------------------------------------------------------#
#INPUT: Pos ngram matches, as DataFrame ------------------------------------------------------#
#OUTPUT: Dictionary with ngrams and their frequency ------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_ngram_count(current_df):
	
	import pandas as pd
	import cytoolz as ct
	
	tuple_list = [tuple(x) for x in current_df.values]
	pair_frequency = ct.frequencies(tuple_list)
	
	return pair_frequency
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#