#---------------------------------------------------------------------------------------------#
#FUNCTION: get_coverage_column ---------------------------------------------------------------#
#INPUT: Full Vector DataFrame ----------------------------------------------------------------#
#OUTPUT: Dataframe with column indicating how many non-sparse features each row has ----------#
#---------------------------------------------------------------------------------------------#
def get_coverage_column(full_vector_df):
	
	import pandas as pd
	
	count_series = (full_vector_df != 0).astype(int).sum(axis=1)
	
	full_vector_df.loc[:,'Coverage'] = count_series
	
	return full_vector_df
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#