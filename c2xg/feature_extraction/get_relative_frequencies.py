#---------------------------------------------------------------------------------------------#
#INPUT: Full feature DataFrame ---------------------------------------------------------------#
#OUTPUT: DataFrame with frequencies relative to total words ----------------------------------#
#---------------------------------------------------------------------------------------------#
def get_relative_frequencies(full_vector):
	
	import pandas as pd

	length_series = full_vector.loc[:,'Length'].copy('deep')
	
	full_vector = full_vector.div(full_vector.loc[:,'Length'], axis=0)
	
	full_vector.loc[:,'Length'] = length_series
	
	return  full_vector
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#