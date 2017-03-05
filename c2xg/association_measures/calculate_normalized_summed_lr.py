#---------------------------------------------------------------------------------------------#
#FUNCTION: calculate_normalized_summed_lr ----------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ------------------------------------#
#OUTPUT: Given Delta-P measure ---------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def calculate_normalized_summed_lr(co_occurrence_list, summed_lr, freq_weighted):
	
	length = len(co_occurrence_list)
	normalized_summed_lr = summed_lr / length
	
	return normalized_summed_lr
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#