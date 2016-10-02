#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_pair_status(pair,
					pair_frequency_dictionary, 
					lr_association_dictionary, 
					rl_association_dictionary,
					catenae_threshold
					):
	
	import cytoolz as ct
	
	l_unit = pair[0]
	r_unit = pair[1]
			
	p1_l_pairs = [(x, y) for (x, y) in list(pair_frequency_dictionary.keys()) if x == l_unit]
	p1_r_pairs = [(x, y) for (x, y) in list(pair_frequency_dictionary.keys()) if y == l_unit]
			
			
	p1_lr_freq = [pair_frequency_dictionary[x] for x in p1_l_pairs]
	p1_rl_freq = [pair_frequency_dictionary[x] for x in p1_r_pairs]
			
	p2_l_pairs = [(x, y) for (x, y) in list(pair_frequency_dictionary.keys()) if x == r_unit]
	p2_r_pairs = [(x, y) for (x, y) in list(pair_frequency_dictionary.keys()) if y == r_unit]
			
	p2_lr_freq = [pair_frequency_dictionary[x] for x in p2_l_pairs]
	p2_rl_freq = [pair_frequency_dictionary[x] for x in p2_r_pairs]
			
	#Calculate variables#
	p1_l_freq = sum(p1_lr_freq)
	p1_r_freq = sum(p1_rl_freq)
	p2_l_freq = sum(p2_lr_freq)
	p2_r_freq = sum(p2_rl_freq)
	
	classifier_output = (p1_l_freq - p1_r_freq) + (p2_r_freq - p2_l_freq)
	
	if classifier_output > catenae_threshold:
		status = "Catenae"
		
	else:
		status = "Non-Catenae"
	
	return status
#---------------------------------------------------------------------------------------------#