#---------------------------------------------------------------------------------------------#
#INPUT: Reduced and sorted DataFrame, per file candidate frequency threshold -----------------#
#OUTPUT: Take template, return list of rows to include in candidate search DataFrame ---------#
#---------------------------------------------------------------------------------------------#
def get_candidate_count(current_df, frequency_threshold_constructions_perfile):
	
	import pandas as pd
	
	current_count = 1
	count = 0
	
	previous_row_list = []
	candidate = []
	row_list = []
	
	count_dictionary = {}
	
	for row in current_df.itertuples():
		
		row_list = row[1:]
		
		if row_list == previous_row_list:
			current_count += 1
			
		else:
			
			if current_count >= frequency_threshold_constructions_perfile:
				
				candidate = previous_row_list
				count = current_count
				count_dictionary[candidate] = count
			
			previous_row_list = row_list
			current_count = 1
	
	del current_count
	del count
	del previous_row_list
	del candidate
	del row_list	
	
	return count_dictionary
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#