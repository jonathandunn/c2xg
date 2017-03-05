#---------------------------------------------------------------------------------------------#
#INPUT: DataFrame and current ngram length ---------------------------------------------------#
#OUTPUT: DataFrame modified for ngram search -------------------------------------------------#
#Prepare DataFrame for pos ngram search in sentence expansion --------------------------------#
#---------------------------------------------------------------------------------------------#
def get_search_df_expansion(original_df, length):

	import pandas as pd

	ordered_columns = ['Sent', 'Pos']
	column_list = []
	
	#First, create initial one-unit dataframe#
	holder_df = original_df.loc[:,['Mas', 'Sent', 'Pos']]
	column_list.append(holder_df)
	
	del holder_df
	
	#Second, if additional units are required, add sequentially#
	if length > 1:
	
		for i in range(2,length):
			ordered_columns.append(['Sent', 'Pos'])
		
		for i in range(len(ordered_columns)):
			holder_df = original_df.loc[:,ordered_columns[i]]
			column_list.append(holder_df.shift(-i))
			del holder_df
		
	original_df = pd.concat(column_list, axis=1)
	del column_list
		
	column_names = original_df.columns.values.tolist()
	column_names_new = []
	sent_counter = 1
	pos_counter = 1
		
	for column in column_names:
		if column == 'Sent':
			column_string = 'Sent' + str(sent_counter)
			column_names_new.append(column_string)
			sent_counter += 1
				
		elif column == 'Pos':
			column_string = 'Pos' + str(pos_counter)
			column_names_new.append(column_string)
			pos_counter += 1
				
	column_names_new.insert(0, 'Mas')	
	original_df.columns = column_names_new
	
	return original_df
#---------------------------------------------------------------------------------------------#