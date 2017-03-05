#---------------------------------------------------------------------------------------------#
#FUNCTION: calculate_summed_rl ---------------------------------------------------------------#
#INPUT: DataFrame with pairwise co-occurrence frequencies ------------------------------------#
#OUTPUT: Given Delta-P measure ---------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def calculate_summed_rl(co_occurrence_list, freq_weighted):
	
	summed_rl = 0.0
	lowest_pairwise = ''

	for pair in co_occurrence_list:
		
		a = pair[0]
		b = pair[1]
		c = pair[2]
		d = pair[3]
		
		pair_rl = float(a / (a + b)) - float(c / (c + d))
		
		if freq_weighted == True:
			pair_rl = pair_rl * a
		
		#If threshold, then add or flag accordingly#
		if lowest_pairwise == '':
			lowest_pairwise = pair_rl
			
		elif pair_rl < lowest_pairwise:
			lowest_pairwise = pair_rl
				
		summed_rl += pair_rl
			
	return (summed_rl, lowest_pairwise)
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#