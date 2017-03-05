#---------------------------------------------------------------------------------------------#
#INPUT: template -----------------------------------------------------------------------------#
#OUTPUT: Take a dataframe, the column to repeat, and a listof times to repeat ----------------#
#Specific to creating alt / sent dataframes b/c more efficient than a generalized version ----#
#---------------------------------------------------------------------------------------------#
def create_shifted_template_df(original_df, ordered_columns):
	
	import pandas as pd

	ordered_columns[0] = 'Sent'
	column_list = []
	
	for i in range(len(ordered_columns)):
		holder_df = original_df.loc[:,ordered_columns[i]]
		column_list.append(holder_df.shift(-i))
		del holder_df
	
	original_df = pd.concat(column_list, axis=1)
	del column_list
	
	return original_df
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#