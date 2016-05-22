#---------------------------------------------------------------------------------------------#
#INPUT: List of (possibly duplicated) sentences marking each occurrence of current feature ---#
#OUTPUT: Series with relative frequency for current construction -----------------------------#
#---------------------------------------------------------------------------------------------#
def get_vector_column(current_sentences, number_of_sentences):
	
	import pandas as pd
	import cytoolz as ct
	import numpy as np

	series_list = []
	
	frequency_index = ct.frequencies(current_sentences)
	
	for i in range(1, number_of_sentences+1):
	
		try:
			series_list.append(frequency_index[i])
			
		except:
			series_list.append(0)
	
	return  series_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#