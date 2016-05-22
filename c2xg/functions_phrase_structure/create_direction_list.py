#---------------------------------------------------------------------------------------------#
#INPUT: Dictionary of head directions, current direction, and pos indexes --------------------#
#OUTPUT: List of phrase heads for current direction ------------------------------------------#
#---------------------------------------------------------------------------------------------#
def create_direction_list(head_direction_list, direction, pos_list):

	direction_list = []
	
	for pos_tuple in head_direction_list:
		
		if pos_tuple[1] == direction:
			temp_index = pos_list.index(pos_tuple[0])
			direction_list.append(temp_index)

	return direction_list
#---------------------------------------------------------------------------------------------#