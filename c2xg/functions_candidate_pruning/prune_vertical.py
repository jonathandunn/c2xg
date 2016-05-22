#---------------------------------------------------------------------------------------------#
#INPUT: Candidate vector dataframe pruned by association strength and horizontally -----------#
#OUTPUT: Candidate vector dataframe pruned vertically ----------------------------------------#
#---------------------------------------------------------------------------------------------#
def prune_vertical(full_vector_dataframe):
    
	import pandas as pd
	import time
	
	start_all = time.time()
	
	column_list = full_vector_dataframe.columns.values.tolist()
	column_list = column_list[1:]
	
	
	pruned_vector_dataframe = full_vector_dataframe.drop_duplicates(subset=column_list, keep="first")
	
	end_all = time.time()
	print("Candidates pruned vertically: " + str(end_all - start_all))
	print("Original: " + str(len(full_vector_dataframe)))
	print("Pruned: " + str(len(pruned_vector_dataframe)))
	print("")
	
	return pruned_vector_dataframe
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#