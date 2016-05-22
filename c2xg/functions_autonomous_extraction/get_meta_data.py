#---------------------------------------------------------------------------------------------#
#INPUT: Full Vector DataFrame, and MetaData as (ID, DICTIONARY) tuples -----------------------#
#OUTPUT: Dataframe with meta-data columns added per text -------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_meta_data(full_vector_df, metadata_tuples):
	
	import pandas as pd
	import numpy as np
	
	#Get fields for saved meta-data#
	temp_dictionary = metadata_tuples[0][1]	
	metadata_columns = list(temp_dictionary.keys())
	
	#Initialize dictionary of lists for field values#
	column_dictionary = {}
	
	for column_name in metadata_columns:
		column_dictionary[column_name] = []
	
	#For each vector (e.g., text), get its meta-data, add to lists#
	for text_id in metadata_tuples:

		current_dictionary = text_id[1]
		
		for column_name in list(current_dictionary.keys()):
			column_dictionary[column_name].append(current_dictionary[column_name])
			
	#For each field, create series and add to vector DataFrame#
	for column_name in list(column_dictionary.keys()):

		temp_series = pd.Series(column_dictionary[column_name], name = column_name)
		temp_series.index = np.arange(1, len(temp_series) + 1)
		full_vector_df = pd.concat([full_vector_df, temp_series], axis = 1)

		del temp_series		
	
	return full_vector_df
#---------------------------------------------------------------------------------------------#