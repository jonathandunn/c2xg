#---------------------------------------------------------------------------------------------#
#FUNCTION: process_unitwise_feature_vector ------------------------------------------------------------#
#INPUT: Single candidate, expanded data files, number of total units -------------------------#
#OUTPUT: List of features for current candidate ----------------------------------------------#
#---------------------------------------------------------------------------------------------#
def process_unitwise_feature_vector(candidate_info_list, 
										candidate_frequency_dict, 
										lemma_frequency, 
										lemma_list, 
										pos_frequency, 
										pos_list, 
										category_frequency, 
										category_list, 
										total_units
										):
	
	import pandas as pd
	from functions_candidate_evaluation.get_unitwise_abcd import get_unitwise_abcd
	from functions_candidate_evaluation.calculate_summed_lr import calculate_summed_lr
	from functions_candidate_evaluation.calculate_summed_rl import calculate_summed_rl
	
	vector_list = []
	
	candidate_id = candidate_info_list[0]
	candidate_length = candidate_info_list[1]
	candidate_frequency = candidate_info_list[2]
	full_candidate_str = str(candidate_id)
		
	#NOT PAIRWISE#
	# 11. Beginning-Divided ΔP, Left-to-Right#
	# 12. Beginning-Divided ΔP, Right-to-Left#
	# 13. End-Divided ΔP, Left-to-Right#
	# 14. End-Divided ΔP, Right-to-Left#
	
	if candidate_length < 3:
	
		divided_beginning_lr = 0
		divided_beginning_rl = 0
		divided_end_lr = 0
		divided_end_rl = 0
		
	else:
		
		#Break candidate into appropriate chunks#
		beginning_divided = [[candidate_id[0]], candidate_id[1:]]
		end_divided = [candidate_id[0:(len(candidate_id) -1)], [candidate_id[-1]]]
		
		beginning_list = get_unitwise_abcd(candidate_frequency, 
											beginning_divided, 
											candidate_frequency_dict, 
											lemma_frequency, 
											lemma_list, 
											pos_frequency, 
											pos_list, 
											category_frequency, 
											category_list, 
											total_units
											)
											
		end_list = get_unitwise_abcd(candidate_frequency, 
										end_divided, 
										candidate_frequency_dict, 
										lemma_frequency, 
										lemma_list, 
										pos_frequency, 
										pos_list, 
										category_frequency, 
										category_list, 
										total_units
										)
		
		if beginning_list == [[0,0,0,0]]:
			divided_beginning_lr = 0
			divided_beginning_rl = 0
			
		else:
			divided_beginning_lr_temp = calculate_summed_lr(beginning_list)
			divided_beginning_lr = divided_beginning_lr_temp[0]
			
			divided_beginning_rl_temp = calculate_summed_rl(beginning_list)
			divided_beginning_rl = divided_beginning_rl_temp[0]
		
		if end_list == [[0,0,0,0]]:
		
			divided_end_lr = 0
			divided_end_rl = 0
			
		else:
			divided_end_lr_temp = calculate_summed_lr(end_list)
			divided_end_lr = divided_end_lr_temp[0]
			
			divided_end_rl_temp = calculate_summed_rl(end_list)
			divided_end_rl = divided_end_rl_temp[0]
	
	vector_list	= [full_candidate_str, divided_beginning_lr, divided_beginning_rl, divided_end_lr, divided_end_rl]
	
	return vector_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#