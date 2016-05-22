#---------------------------------------------------------------------------------------------#
#FUNCTION: get_frequency_dict: ---------------------------------------------------------------#
#INPUT: list of all candidates ---------------------------------------------------------------#
#OUTPUT: Dictionary with frequency of each candidate -----------------------------------------#
# Take full candidate list and return frequency dictionary -----------------------------------#
#---------------------------------------------------------------------------------------------#
def get_frequency_dict(full_candidate_list):
	
	candidate_frequency_dict = {}
	
	for i in range(len(full_candidate_list)):
		
		candidate_id = str(full_candidate_list[i][0])
		candidate_length = full_candidate_list[i][1]
		candidate_frequency = full_candidate_list[i][2]
		
		candidate_frequency_dict[candidate_id] = candidate_frequency
			
	return candidate_frequency_dict
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#