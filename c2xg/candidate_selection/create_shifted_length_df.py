#---------------------------------------------------------------------------------------------#
#FUNCTION: create_shifted_length_df ----------------------------------------------------------#
#INPUT: template -----------------------------------------------------------------------------#
#OUTPUT: Take a dataframe, the column to repeat, and a listof times to repeat ----------------#
#Specific to creating alt / sent dataframes b/c more efficient than a generalized version ----#
#---------------------------------------------------------------------------------------------#
def create_shifted_length_df(original_df, current_length):
	
	import pandas as pd

	column_list = []
	
	ordered_columns = []
	named_columns = []
	
	ordered_columns.append(['Sent', 'Mas'])
	named_columns.append('Sent')
	named_columns.append('Mas')
	
	for i in range(current_length):
		ordered_columns.append(["Lex", 'Pos', 'Cat'])
		named_columns.append("Lex" + str(i))
		named_columns.append('Pos' + str(i))
		named_columns.append('Cat' + str(i))
		
	for i in range(len(ordered_columns)):
		holder_df = original_df.loc[:,ordered_columns[i]]
		column_list.append(holder_df.shift(-i))
		del holder_df

		if i == len(ordered_columns)-1:
			holder_df = original_df.loc[:,"Mas"]
			column_list.append(holder_df.shift(-i))
			named_columns.append('EndMas')
			
	
	original_df = pd.concat(column_list, axis=1)
	del column_list
	
	original_df.columns = named_columns
	
	return original_df
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#