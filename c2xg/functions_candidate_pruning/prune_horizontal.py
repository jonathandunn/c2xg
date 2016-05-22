#---------------------------------------------------------------------------------------------#
#INPUT: Candidate vector dataframe pruned by association strength ----------------------------#
#OUTPUT: Candidate vector dataframe pruned horizontally --------------------------------------#
#---------------------------------------------------------------------------------------------#
def prune_horizontal(full_vector_dataframe):
    
	import pandas as pd
	import time
	
	start_all = time.time()
	
	previous_row = []
	duplicate_list = []
	
	candidate_list = full_vector_dataframe.loc[:,'Candidate'].tolist()
	sorted_dataframe_list = []

	for i in range(len(candidate_list)):
		line = eval(candidate_list[i])
		sorted_dataframe_list.append(line)
		
	sorted_dataframe = pd.DataFrame(sorted_dataframe_list)
	
	column_list = sorted_dataframe.columns.values.tolist()
	sorted_dataframe = sorted_dataframe.sort_values(column_list)
	
	for row in sorted_dataframe.itertuples():
	
		row = row[1:]
				
		if previous_row == []:
			previous_row = row
			
		else:

			try:
				end = row.index(None)
				
			except:
				end = len(column_list)
				
			row = row[:end]
			
			if row == previous_row[:len(row)]:

				current_string = "[" + str(row) + "]"
				current_string = current_string.replace("((", "(")
				current_string = current_string.replace("))", ")")
				duplicate_list.append(current_string)
			
			previous_row = row[:end]
	
	del sorted_dataframe
	
	row_mask = full_vector_dataframe.loc[:,'Candidate'].isin(duplicate_list)
	pruned_vector_dataframe = full_vector_dataframe.loc[~row_mask,]
	
	end_all = time.time()
	print("Candidates pruned horizontally: " + str(end_all - start_all))
	print("Original: " + str(len(full_vector_dataframe)))
	print("Pruned: " + str(len(pruned_vector_dataframe)))
	print("")
	
	return pruned_vector_dataframe
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#