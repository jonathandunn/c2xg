#---------------------------------------------------------------------------------------------#
#FUNCTION: get_dictionary --------------------------------------------------------------------#
#INPUT: List of candidates and abcd's --------------------------------------------------------#
#OUTPUT: Dictionary with co-occurrence data (a, b, c) for each key ---------------------------#
#---------------------------------------------------------------------------------------------#
def get_dictionary(pairwise_list):

	pairwise_dictionary = {}
	
	for i in range(len(pairwise_list)):
		
		temp_id = pairwise_list[i][0]
		temp_a = pairwise_list[i][1]
		temp_b = pairwise_list[i][2]
		temp_c = pairwise_list[i][3]
		temp_d = pairwise_list[i][4]
		
		pairwise_dictionary[temp_id] = [temp_a, temp_b, temp_c, temp_d]
	
	return pairwise_dictionary
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#