#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_pair_status_same(pair,
						pair_frequency_dictionary, 
						lr_association_dictionary, 
						rl_association_dictionary,
						same_unit_dictionary
						):

	same_pairs = [(x, y) for (x, y) in list(pair_frequency_dictionary.keys()) if x == y]
	same_pairs_freq = [pair_frequency_dictionary[x] for x in same_pairs]
	
	mean_freq = sum(same_pairs_freq) / float(len(same_pairs_freq))
	
	if pair_frequency_dictionary[pair] > mean_freq:
		status = "Catenae"
		same_unit_dictionary[pair[0]] = "Catenae"
		
	else:	
		status = "Non-Catenae"
		same_unit_dictionary[pair[0]] = "Non-Catenae"
	
	return status, same_unit_dictionary
#---------------------------------------------------------------------------------------------#