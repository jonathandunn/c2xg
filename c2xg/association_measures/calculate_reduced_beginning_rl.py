#---------------------------------------------------------------------------------------------#
#FUNCTION: calculate_reduced_beginning_rl ----------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ------------------------------------#
#OUTPUT: Given Delta-P measure ---------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def calculate_reduced_beginning_rl(co_occurrence_list, freq_weighted):
	
	from association_measures.calculate_summed_rl import calculate_summed_rl
	
	temp_summed = calculate_summed_rl(co_occurrence_list, freq_weighted)
	main_summed = float(temp_summed[0])
	
	temp_reduced = calculate_summed_rl(co_occurrence_list[1:], freq_weighted)
	beginning_reduced_summed = float(temp_reduced[0])
	
	beginning_reduced_rl = main_summed - beginning_reduced_summed
	
	return beginning_reduced_rl
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#