#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def calculate_endpoint(co_occurrence_list, pairwise_dictionary, candidate_id, freq_weighted):
	
	from association_measures.calculate_summed_lr import calculate_summed_lr
	from association_measures.calculate_summed_rl import calculate_summed_rl
	import cytoolz as ct
	
	unit1 = str(candidate_id[0])
	unit2 = str(candidate_id[len(candidate_id)-1])
	
	a = 0
	b = 0
	c = 0
	d = 0
	
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
		
	if a == 0:
		lr_measure = 0.0
		rl_measure = 0.0
		
	else:
		
		lr_measure = calculate_summed_lr([[a, b, c, d]], freq_weighted)
		lr_measure = lr_measure[0]
		
		rl_measure = calculate_summed_rl([[a, b, c, d]], freq_weighted)
		rl_measure = rl_measure[0]

	return lr_measure, rl_measure
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#