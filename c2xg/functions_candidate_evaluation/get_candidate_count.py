#-------------------------------------------#
#-- Count candidates in dict ---------------#
#-------------------------------------------#
def get_candidate_count(candidate_dict):

	total = 0
	
	for key in candidate_dict.keys():
		total += len(list(candidate_dict[key].keys()))
		
	return total
#--------------------------------------------#