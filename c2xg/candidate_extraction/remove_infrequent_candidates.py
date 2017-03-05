#---------------------------------------------------------------------------------------------#
#INPUT: Current template and data_file_candidate_constructions -------------------------------#
#OUTPUT: Filename for storing candidates from current template -------------------------------#
#---------------------------------------------------------------------------------------------#
def remove_infrequent_candidates(candidate_list, data_file_candidate_constructions):
    
	for i in range(len(candidate_list)):
		if candidate_dictionary[i] > frequency_threshold_constructions:
			
			temp_list = candidate_list[i]
			final_candidate_list.append(temp_list)
			
	print("")
	print("For template: " + str(template) + ", All candidates: " + str(len(candidate_list)))
	print("For template: " + str(template) + ", Reduced candidates: " + str(len(final_candidate_list)))
		
	return pickled_list_file
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#