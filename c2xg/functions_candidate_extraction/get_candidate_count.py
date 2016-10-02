#---------------------------------------------------------------------------------------------#
#INPUT: Reduced and sorted DataFrame, per file candidate frequency threshold -----------------#
#OUTPUT: Take template, return list of rows to include in candidate search DataFrame ---------#
#---------------------------------------------------------------------------------------------#
def get_candidate_count(current_df, frequency_threshold_constructions_perfile):
	
	import pandas as pd
	import cytoolz as ct

	tuple_list = [tuple(x) for x in current_df.values]
	pair_frequency = ct.frequencies(tuple_list)
	
	above_threshold = lambda x: x > frequency_threshold_constructions_perfile
	pair_frequency = ct.valfilter(above_threshold, pair_frequency)
	
	
	
	return pair_frequency	
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#