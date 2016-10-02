#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_pair_head(pair,
					pair_frequency_dictionary, 
					lr_association_dictionary, 
					rl_association_dictionary
					):

	import cytoolz as ct
	
	l_unit = pair[0]
	r_unit = pair[1]
			
	p1_l_pairs = [(x, y) for (x, y) in list(pair_frequency_dictionary.keys()) if x == l_unit]
	p1_lr_assoc = [lr_association_dictionary[x] for x in p1_l_pairs]
				
	p2_r_pairs = [(x, y) for (x, y) in list(pair_frequency_dictionary.keys()) if y == r_unit]
	p2_rl_assoc = [rl_association_dictionary[x] for x in p2_r_pairs]
			
	#Calculate variables#
	p1_l_sum = sum(p1_lr_assoc)
	p2_r_sum = sum(p2_rl_assoc)
	
	classifier_output = p1_l_sum - p2_r_sum
	
	if classifier_output > 0:
		status = "L"
		
	elif classifier_output < 0:
		status = "R"

	return status
#---------------------------------------------------------------------------------------------#