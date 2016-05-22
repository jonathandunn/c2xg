#---------------------------------------------------------------------------------------------#
#INPUT: Candidate vector dataframe pruned by association strength and horizontally -----------#
#OUTPUT: Candidate vector dataframe ranked by association strength ---------------------------#
#---------------------------------------------------------------------------------------------#
def rank_constructions(full_vector_dataframe):
    
	import pandas as pd
	import time
	
	start_all = time.time()
	
	#First, get column for the greatest normalized summed Delta-P score, either RL or LR#
	full_vector_dataframe['GreatestSummed'] = full_vector_dataframe.apply(lambda row: max(row['Normalized_Summed_LR'], row['Normalized_Summed_RL']), axis=1)
	
	#Second, get column for greatest beginning divided Delta-P score, either RL or LR#
	full_vector_dataframe['GreatestB-Divided'] = full_vector_dataframe.apply(lambda row: max(row['Beginning_Divided_LR'], row['Beginning_Divided_RL']), axis=1)
	
	#Third, get column for greatest end divided Delta-P score, either RL or LR#
	full_vector_dataframe['GreatestE-Divided'] = full_vector_dataframe.apply(lambda row: max(row['End_Divided_LR'], row['End_Divided_RL']), axis=1)
	
	#Fourth, get column for greatest of these Delta-P scores and sort accordingly#
	full_vector_dataframe['Ranking'] = full_vector_dataframe.apply(lambda row: max(row['GreatestSummed'], row['GreatestB-Divided'], row['GreatestE-Divided']), axis=1)
	full_vector_dataframe = full_vector_dataframe.sort_values(by="Ranking", ascending=False)
	
	end_all = time.time()
	print("Candidates ranked: " + str(end_all - start_all))
	print("")

	return full_vector_dataframe
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#