#---------------------------------------------------------------------------------------------#
#-- Take association vectors, threshold formula, and feature; return string of query ---------#
#---------------------------------------------------------------------------------------------#
def calculate_threshold(full_vector_df, 
						threshold_formula,
						relation,
						column_name
						):
    
	import pandas as pd

	mean = full_vector_df.loc[:,column_name].mean()
	std = full_vector_df.loc[:,column_name].std()

	if threshold_formula == "Low":
		threshold = mean - std

	elif threshold_formula == "Medium":
		threshold = mean
	
	elif threshold_formula == "High":
		threshold = mean + std
		
	else:
		print("")
		print("!!PROBLEM!!!")
		print(threshold_formula)
		print("")
	
	query_string = "(" + str(column_name) + " " + str(relation) + " " + str(threshold) + ")"
	
	return query_string
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#