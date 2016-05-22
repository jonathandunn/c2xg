#---------------------------------------------------------------------------------------------#
#INPUT: template -----------------------------------------------------------------------------#
#OUTPUT: Take template, return list of rows to include in candidate search DataFrame ---------#
#---------------------------------------------------------------------------------------------#
def create_shifted_df(original_df, desired_column, ordered_columns):
	
	import pandas as pd

	holder_df = original_df.iloc[:,desired_column]
	column_dict = {}
	
	for i in range(len(ordered_columns)):	
		column_dict[i] = holder_df.shift(-i)

	original_df = pd.DataFrame(column_dict)	
	del column_dict
	del holder_df

	return original_df
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#