#---------------------------------------------------------------------------------------------#
#INPUT: Length of current n-gram window and direction flag, pos index to start with ----------#
#OUTPUT: String of query ---------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_expansion_query(constituent):

	if len(constituent) > 1:
	
		query = "(Pos1 == " + str(constituent[0])
	
		for i in range(1,len(constituent)):
			query += " and Pos" + str(i+1) + " == " + str(constituent[i])
		
		query += ") and (Sent1 "
	
		for i in range(1, len(constituent)):
			query += "== Sent" + str(i)
		
		query += ")"
	
	elif len(constituent) == 1:
		query = "(Pos1 == " + str(constituent[0]) + ")"
	
	return query
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#