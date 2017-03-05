#---------------------------------------------------------------------------------------------#
#FUNCTION: calculate_reduced_end_lr ----------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ------------------------------------#
#OUTPUT: Given Delta-P measure ---------------------------------------------------------------#
def calculate_reduced_end_lr(co_occurrence_list, freq_weighted):
	
	from association_measures.calculate_summed_lr import calculate_summed_lr
	
	length = len(co_occurrence_list) - 1
	
	temp_summed = calculate_summed_lr(co_occurrence_list, freq_weighted)
	main_summed = float(temp_summed[0])
	
	temp_end = calculate_summed_lr(co_occurrence_list[:length], freq_weighted)
	end_reduced_summed = float(temp_end[0])
	
	end_reduced_lr = main_summed - end_reduced_summed
	
	return end_reduced_lr
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#