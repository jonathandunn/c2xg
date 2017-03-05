#---------------------------------------------------------------------------------------------#
#FUNCTION: create_pairwise_single ------------------------------------------------------------#
#INPUT: Single candidate, expanded data files, number of total units -------------------------#
#OUTPUT: List of features for current candidate ----------------------------------------------#
#---------------------------------------------------------------------------------------------#
def create_pairwise_single(candidate_id, 
							candidate_frequency, 
							pairwise_dictionary,
							freq_weighted
							):
	
	import pandas as pd
	import cytoolz as ct
	
	from association_measures.calculate_summed_lr import calculate_summed_lr
	from association_measures.calculate_summed_rl import calculate_summed_rl
	
	from association_measures.calculate_normalized_summed_lr import calculate_normalized_summed_lr
	from association_measures.calculate_normalized_summed_rl import calculate_normalized_summed_rl
	
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
	candidate_str = str(candidate_id)
	candidate_vector = [candidate_str, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	
	try:
		current_pair = ct.get(candidate_str, pairwise_dictionary)
	except:
		current_pair = []
	
		
	if current_pair !=[]:
	
		a = current_pair[0]
		b = current_pair[1]
		c = current_pair[2]
		d = current_pair[3]
			
		co_occurrence_list = [[a, b, c, d]]

		#Second, calculate Delta P's for pairwise measures#
		lr_tuple = calculate_summed_lr(co_occurrence_list, freq_weighted)
		summed_lr = lr_tuple[0]
		smallest_lr = lr_tuple[1]
		
		rl_tuple = calculate_summed_rl(co_occurrence_list, freq_weighted)
		summed_rl = rl_tuple[0]
		smallest_rl = rl_tuple[1]
	
		normalized_summed_lr = calculate_normalized_summed_lr(co_occurrence_list, summed_lr, freq_weighted)
		normalized_summed_rl = calculate_normalized_summed_rl(co_occurrence_list, summed_rl, freq_weighted)
	
		end_reduced_lr = 0
		end_reduced_rl = 0
	
		beginning_reduced_lr = 0
		beginning_reduced_rl = 0
	
		directional_scalar = 0
		directional_categorical = 0
	
		#Third, create list of feature values for current candidate, including candidate id#
	
		candidate_vector = [candidate_str, 
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