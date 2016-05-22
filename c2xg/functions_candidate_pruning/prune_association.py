#---------------------------------------------------------------------------------------------#
#INPUT: Full candidate vector dataframe ------------------------------------------------------#
#OUTPUT: Candidate vector dataframe pruned by association strength ---------------------------#
#---------------------------------------------------------------------------------------------#
def prune_association(full_vector_dataframe, 
						pairwise_threshold_lr, 
						pairwise_threshold_rl
						):
    
	import pandas as pd
	import time
	
	start_all = time.time()
	
	#First, meet all pairwise minimum thresholds; remove values below thresholds#
	#Thresholds are relative to total distribution, so first find mean and standard deviations#
	smallest_lr_mean = full_vector_dataframe.loc[:,'Smallest_LR'].mean()
	smallest_lr_std = full_vector_dataframe.loc[:,'Smallest_LR'].std()
	lr_threshold = smallest_lr_mean + (smallest_lr_std * pairwise_threshold_lr)
	
	smallest_rl_mean = full_vector_dataframe.loc[:,'Smallest_RL'].mean()
	smallest_rl_std = full_vector_dataframe.loc[:,'Smallest_RL'].std()
	rl_threshold = smallest_rl_mean + (smallest_rl_std * pairwise_threshold_rl)
	
	query_string = "(Smallest_LR >= @lr_threshold or Smallest_RL >= @rl_threshold)"
	
	#Second, candidate doesn't lose association strength when reduced#
	#This means that the beginning and end reduced measures are positive#
	#Single pair units are set to 0, so only values below 0#
	
	query_string += " and (Beginning_Reduced_LR >= 0 and Beginning_Reduced_RL >= 0 and End_Reduced_LR >= 0 and End_Reduced_RL >= 0)"
	
	#Third, there are no changes in direction of association#
	#This means that the directional categorical measure is 0#
	
	query_string += " and (Directional_Categorical == 0)"
	
	pruned_vector_dataframe = full_vector_dataframe.query(query_string, parser='pandas', engine='numexpr')
	
	end_all = time.time()
	print("Candidates pruned by association strength: " + str(end_all - start_all))
	print("Original: " + str(len(full_vector_dataframe)))
	print("Pruned: " + str(len(pruned_vector_dataframe)))
	print("")
	
	return pruned_vector_dataframe
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#