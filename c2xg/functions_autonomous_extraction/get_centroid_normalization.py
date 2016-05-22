#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_centroid_normalization(full_vector_df, centroid_df):
	
	import pandas as pd
	import numpy as np
	
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