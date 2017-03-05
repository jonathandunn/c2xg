#---------------------------------------------------------------------------------------------#
#FUNCTION: get_pair_list: --------------------------------------------------------------------#
#INPUT: list of all candidates ---------------------------------------------------------------#
#OUTPUT: List of all pairs -------------------------------------------------------------------#
# Take full candidate list and return all two item pairs -----#
#---------------------------------------------------------------------------------------------#
def get_formatted_candidates(full_candidate_dictionary):
	
	candidate_list_formatted = []
	candidate_list_all = []
	candidate_list_pairs = []
	
	for key in full_candidate_dictionary.keys():
		
		current_template = eval(key)
		current_dictionary = full_candidate_dictionary[key]
		current_candidate_list = list(current_dictionary.keys())
		
		for j in range(len(current_candidate_list)):
			
			current_candidate = current_candidate_list[j]
			formatted_candidate = []
			candidate_length = 0
			
			for k in range(len(current_candidate)):
				temp_tuple = (current_template[k], current_candidate[k])
				formatted_candidate.append(temp_tuple)
				candidate_length += 1
			
			candidate_list_formatted.append(formatted_candidate)
			candidate_frequency = current_dictionary[current_candidate_list[j]]
			
			candidate_list_all.append([formatted_candidate, candidate_length, candidate_frequency])
			
			if candidate_length == 2:
				candidate_list_pairs.append([formatted_candidate, candidate_length, candidate_frequency])
				
	return [candidate_list_formatted, candidate_list_all, candidate_list_pairs]
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#