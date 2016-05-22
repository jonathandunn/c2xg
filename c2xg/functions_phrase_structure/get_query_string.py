#---------------------------------------------------------------------------------------------#
#INPUT: Length of current n-gram window and direction flag, pos index to start with ----------#
#OUTPUT: String of query ---------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_query_string(length, pos_index, direction_flag):
		
	query = "(Sent1 == Sent2"
	
	for i in range(2, length):
	
		query += " == Sent" + str(i+1)
		
	query += ")"
	
	query += " & (Pos1 != 0 & Pos2 !=0"
	
	
	for i in range(2, length):
	
		query += " & Pos" + str(i+1) + " != 0"
		
	query += ") "
	
	if direction_flag == "LR":
	
		query += " & (Pos1 == " + str(pos_index) +")"
		
	elif direction_flag == "RL":
	
		query += " & (Pos" + str(length) + " == " + str(pos_index) +")"
	
	return query
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#