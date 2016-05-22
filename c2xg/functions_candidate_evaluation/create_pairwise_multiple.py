#---------------------------------------------------------------------------------------------#
#FUNCTION: create_pairwise_multiple ----------------------------------------------------------#
#INPUT: Single candidate, expanded data files, number of total units -------------------------#
#OUTPUT: List of features for current candidate ----------------------------------------------#
#---------------------------------------------------------------------------------------------#
def create_pairwise_multiple(candidate_id, 
								candidate_frequency, 
								pairwise_dictionary
								):
	
	import pandas as pd
	import cytoolz as ct
	
	from functions_candidate_evaluation.calculate_summed_lr import calculate_summed_lr
	from functions_candidate_evaluation.calculate_summed_rl import calculate_summed_rl
	
	from functions_candidate_evaluation.calculate_normalized_summed_lr import calculate_normalized_summed_lr
	from functions_candidate_evaluation.calculate_normalized_summed_rl import calculate_normalized_summed_rl
	
	from functions_candidate_evaluation.calculate_reduced_beginning_lr import calculate_reduced_beginning_lr
	from functions_candidate_evaluation.calculate_reduced_beginning_rl import calculate_reduced_beginning_rl
	
	from functions_candidate_evaluation.calculate_reduced_end_lr import calculate_reduced_end_lr
	from functions_candidate_evaluation.calculate_reduced_end_rl import calculate_reduced_end_rl
	
	from functions_candidate_evaluation.calculate_directional_scalar import calculate_directional_scalar
	from functions_candidate_evaluation.calculate_directional_categorical import calculate_directional_categorical

	full_candidate_str = str(candidate_id)
	candidate_vector = [full_candidate_str, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	co_occurrence_list = []
	
	# Features for Vector:#
		#PAIRWISE#
		# 1. Simple Frequency (Relative in the sense that all candidates are in same corpus)#
		# 2. Summed ΔP, Left-to-Right#
		# 3. Smallest Pairwise LR
		# 4. Summed ΔP, Right-to-Left#
		# 5. Smallest Pairwise RL
		# 6. Normalized (Summed ΔP, Left-to-Right)#
		# 7. Normalized (Summed ΔP, Right-to-Left)#
		# 8. Beginning-Reduced ΔP, Left-to-Right#
		# 9. Beginning-Reduced ΔP, Right-to-Left#
		# 10. End-Reduced ΔP, Left-to-Right#
		# 11. End-Reduced ΔP, Right-to-Left#
		# 12. Directional ΔP#
	
	#First, get A, B, C, D for pair from pairwise_df#
	for i in range(len(candidate_id) - 1):
		
		unit1 = str(candidate_id[i])
		unit2 = str(candidate_id[i+1])
	
		candidate_str = "[" + unit1 + ", " + unit2 + "]"
				
		try:
			current_pair = ct.get(candidate_str, pairwise_dictionary)
		except:
			current_pair = []
			
		if current_pair !=[]:
		
			a = current_pair[0]
			b = current_pair[1]
			c = current_pair[2]
			d = current_pair[3]
			
			co_occurrence_list.append([a, b, c, d])
	
	if len(co_occurrence_list) > 1:

		#Second, calculate Delta P's for pairwise measures#
		lr_tuple = calculate_summed_lr(co_occurrence_list)
		summed_lr = lr_tuple[0]
		smallest_lr = lr_tuple[1]
		
		rl_tuple = calculate_summed_rl(co_occurrence_list)
		summed_rl = rl_tuple[0]
		smallest_rl = rl_tuple[1]
	
		normalized_summed_lr = calculate_normalized_summed_lr(co_occurrence_list, summed_lr)
		normalized_summed_rl = calculate_normalized_summed_rl(co_occurrence_list, summed_rl)
	
		end_reduced_lr = calculate_reduced_end_lr(co_occurrence_list)
		end_reduced_rl = calculate_reduced_end_rl(co_occurrence_list)
	
		beginning_reduced_lr = calculate_reduced_beginning_lr(co_occurrence_list)
		beginning_reduced_rl = calculate_reduced_beginning_rl(co_occurrence_list)
	
		directional_scalar = calculate_directional_scalar(co_occurrence_list)
		directional_categorical = calculate_directional_categorical(co_occurrence_list)
	
		#Third, create list of feature values for current candidate, including candidate id#
	
		candidate_vector = [full_candidate_str, 
							candidate_frequency, 
							summed_lr, 
							smallest_lr,
							summed_rl, 
							smallest_rl,
							normalized_summed_lr, 
							normalized_summed_rl, 
							beginning_reduced_lr,
							beginning_reduced_rl,
							end_reduced_lr,
							end_reduced_rl,
							directional_scalar,
							directional_categorical
						]
	
	return candidate_vector
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#