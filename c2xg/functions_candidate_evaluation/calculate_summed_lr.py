#---------------------------------------------------------------------------------------------#
#FUNCTION: calculate_summed_lr ---------------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ------------------------------------#
#OUTPUT: Given Delta-P measure ---------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def calculate_summed_lr(co_occurrence_list, freq_weighted):
	
	summed_lr = 0.0
	lowest_pairwise = ''

	for i in range(len(co_occurrence_list)):

		a = float(co_occurrence_list[i][0])
		b = float(co_occurrence_list[i][1])
		c = float(co_occurrence_list[i][2])
		d = float(co_occurrence_list[i][3])
		
		pair_lr = float(a / (a + c)) - float(b / (b + d))
		
		if freq_weighted == True:
			pair_lr = pair_lr * a
		
		#If threshold, then add or flag accordingly#
		if lowest_pairwise == '':
			lowest_pairwise = pair_lr
			
		elif pair_lr < lowest_pairwise:
			lowest_pairwise = pair_lr
				
		summed_lr += pair_lr
			
	return (summed_lr, lowest_pairwise)
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#