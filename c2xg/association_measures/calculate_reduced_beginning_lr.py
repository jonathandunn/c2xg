#---------------------------------------------------------------------------------------------#
#FUNCTION: calculate_reduced_beginning_lr ----------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ------------------------------------#
#OUTPUT: Given Delta-P measure ---------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def calculate_reduced_beginning_lr(co_occurrence_list, freq_weighted):
	
	from association_measures.calculate_summed_lr import calculate_summed_lr
	
	temp_summed = calculate_summed_lr(co_occurrence_list, freq_weighted)
	main_summed = float(temp_summed[0])
	
	temp_beginning = calculate_summed_lr(co_occurrence_list[1:], freq_weighted)
	beginning_reduced_summed = float(temp_beginning[0])
	
	beginning_reduced_lr = main_summed - beginning_reduced_summed
	
	return beginning_reduced_lr
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#