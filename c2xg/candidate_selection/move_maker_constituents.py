#------------------------------------------------------------------------------#
def move_maker_constituents(current_head_dictionary, tabu_list, best_move):

	head_list = []
	for move in best_move:
	
		move_head = move[0]
		move_type = move[1]
		head_list.append(move_head)
		
		current_head_dictionary[move_head][move_type] = current_head_dictionary[move_head][move_type] * -1
	
	tabu_list.appendleft(head_list)

	return current_head_dictionary, tabu_list
#------------------------------------------------------------------------------#