#---------------------------------------------------------------------------------------------#
#FUNCTION: calculate_reduced_end_rl ----------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ------------------------------------#
#OUTPUT: Given Delta-P measure ---------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def calculate_reduced_end_rl(co_occurrence_list):
	
	from functions_candidate_evaluation.calculate_summed_rl import calculate_summed_rl
	
	length = len(co_occurrence_list) - 1
	
	temp_summed = calculate_summed_rl(co_occurrence_list)
	main_summed = float(temp_summed[0])
	
	temp_end = calculate_summed_rl(co_occurrence_list[:length])
	end_reduced_summed = float(temp_end[0])
	
	end_reduced_rl = main_summed - end_reduced_summed
	
	return end_reduced_rl
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#