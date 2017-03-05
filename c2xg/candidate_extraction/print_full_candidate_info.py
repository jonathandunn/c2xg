#---------------------------------------------------------------------------------------------#
#INPUT: Full candidate list ------------------------------------------------------------------#
#OUTPUT: None: just print how many candidates total ------------------------------------------#
#---------------------------------------------------------------------------------------------#
def print_full_candidate_info(full_candidate_list):

	total_count = 0
	
	for item in full_candidate_list:
	
		template = item[0]
		dictionary = item[1]
		
		current_count = len(dictionary.keys())
		total_count += current_count
		
	print("Total number of frequency reduced candidates: " + str(total_count))
	
	return
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#