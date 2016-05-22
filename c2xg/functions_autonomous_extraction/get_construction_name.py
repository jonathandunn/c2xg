#---------------------------------------------------------------------------------------------#
#FUNCTION: get_construction_name -------------------------------------------------------------#
#INPUT: Candidate construction and index lists -----------------------------------------------#
#OUTPUT: String of readable construction id --------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_construction_name(candidate, 
							lemma_list, 
							pos_list, 
							category_list
							):
	
	candidate_id = ""
	
	for i in range(len(candidate)):
	
		current_col = candidate[i][0]
		current_index = candidate[i][1]
		
		if current_col == "Lem":
			index_list = lemma_list
		
		elif current_col == "Pos":
			index_list = pos_list
		
		elif current_col == "Cat":
			index_list = category_list
			
		current_unit = index_list[current_index]
		
		candidate_id += str(current_col) + ":" + str(current_unit) + " "
		
	candidate_id = candidate_id.encode()
		
	return candidate_id
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#