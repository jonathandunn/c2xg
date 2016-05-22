#---------------------------------------------------------------------------------------------#
#INPUT: Pos ngram matches, as DataFrame ------------------------------------------------------#
#OUTPUT: Dictionary with ngrams and their frequency ------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_ngram_count(current_df):
	
	import pandas as pd
	
	current_count = 1
	count = 0
	
	previous_row_list = ""
	candidate = []
	row_list = []
	
	count_dictionary = {}
	
	for row in current_df.itertuples():
		
		row_list = row[1:]
		row_list = str(row_list)
		
		if row_list == previous_row_list:
			current_count += 1
			
		else:
			
			candidate = previous_row_list
			count = current_count
			count_dictionary[candidate] = count
			
			previous_row_list = row_list
			current_count = 1
	
	return count_dictionary
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#