#---------------------------------------------------------------------------------------------#
#INPUT: Pos list, lists of heads in both directions ------------------------------------------#
#OUTPUT: Updated pos_list with new phrase type indexes added ---------------------------------#
#Take pos list and return updated pos list with phrase types added ---------------------------#
#---------------------------------------------------------------------------------------------#
def update_pos_list(pos_list, phrase_independence_list):

	#There are two types of head: dependent and independent#
	#Dependent heads cannot occur on their own  (e.g., "the" is not a DT_phrase)
	#Independent heads can occur on their own (e.g., "dog" can be an NN_phrase)
	#Dependent heads need a pos label distinct from individual units of that type#
	
	for pos_tuple in phrase_independence_list:
		
		current_pos = pos_tuple[0]
		current_status = pos_tuple[1]
		
		if current_status == "Dependent":
			current_addition = current_pos + "_PHRASE"
			pos_list.append(current_addition)		
	
	pos_list.append("MNE")
	
	return pos_list
#---------------------------------------------------------------------------------------------#