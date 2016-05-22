#---------------------------------------------------------------------------------------------#
#FUNCTION: get_query_autonomous_candidate ----------------------------------------------------#
#INPUT: List of columns to check for equivalence ---------------------------------------------#
#OUTPUT: String of query ---------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_query_autonomous_candidate(current_candidate):
		
	query = ""
	
	for i in range(len(current_candidate)):
	
		current_col = current_candidate[i][0]
		current_index = current_candidate[i][1]
		
		if i == 0:
			query = "(" + str(current_col) + str(i) + " == " + str(current_index) + " "
			
		elif i  > 0:
			query += "and " + str(current_col) + str(i) + " == " + str(current_index) + " "
	
	query += ")"
	
	return query
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#