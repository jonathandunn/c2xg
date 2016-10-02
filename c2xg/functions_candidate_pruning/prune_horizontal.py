#---------------------------------------------------------------------------------------------#
#INPUT: Candidate vector dataframe pruned by association strength ----------------------------#
#OUTPUT: Candidate vector dataframe pruned horizontally --------------------------------------#
#---------------------------------------------------------------------------------------------#
def prune_horizontal(full_vector_df):
    
	import pandas as pd
	
	delete_list = []
	
	sort_df = full_vector_df.loc[:, ["Candidate", "End_Divded_RL"]]
	sort_df = sort_df.sort_values(by = "Candidate", ascending = False, inplace = False)
		
	last = sort_df.iloc[0]
	
	for i in range(1, sort_df.shape[0]):
		
		current = sort_df.iloc[i]
		
		current_candidate = str(current[0])
		current_candidate = current_candidate[1:len(current_candidate)-1]
		
		last_candidate = str(last[0])
		last_candidate = last_candidate[1:len(last_candidate)-1]
		
		
		if current_candidate in last_candidate:
			delete_list.append(current.name)
		
		last = sort_df.iloc[i]
			
	if len(delete_list) > 0:	
		pruned_vector_df = full_vector_df[~full_vector_df.index.isin(delete_list)]
	
	else:
		pruned_vector_df = full_vector_df

	return pruned_vector_df
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#