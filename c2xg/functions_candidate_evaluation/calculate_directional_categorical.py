#---------------------------------------------------------------------------------------------#
#FUNCTION: calculate_directional -------------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ------------------------------------#
#OUTPUT: Given Delta-P measure ---------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def calculate_directional_categorical(co_occurrence_list, freq_weighted):
	
	from functions_candidate_evaluation.calculate_summed_lr import calculate_summed_lr
	from functions_candidate_evaluation.calculate_summed_rl import calculate_summed_rl
	
	lr_dominate = 0
	rl_dominate = 0
	
	for i in range(len(co_occurrence_list) - 1):
	
		pair = co_occurrence_list[i:i+1]
	
		temp_lr = calculate_summed_lr(pair, freq_weighted)
		summed_lr = float(temp_lr[0])
		
		temp_rl = calculate_summed_rl(pair, freq_weighted)
		summed_rl = float(temp_rl[0])
		
		current_difference = summed_lr - summed_rl
		
		if current_difference > 0:
			lr_dominate += 1
			
		else:
			rl_dominate += 1
			
	#Combine pairwise dominance#
	directional_categorical = min(lr_dominate, rl_dominate)
	
	return directional_categorical
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#