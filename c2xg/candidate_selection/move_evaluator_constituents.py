#----------------------------------------------------------------------#
def move_evaluator_constituents(move_size, move_list, sequence_list, head_dictionary, current_score):

	import random
	from candidate_selection.check_constituent_constraints import check_constituent_constraints
	
	#---Randomly change N items from move list------------------------------------------------#
	try:
		current_move_list = random.sample(move_list, move_size)
	
	except:
		current_move_list = move_list

	for move in current_move_list:
	
		head = move[0]
		parameter = move[1]
	
		#Reverse the parameter for the head specified in the current move#
		head_dictionary[head][parameter] = head_dictionary[head][parameter] * -1
	#---Done randomly selecting moves----------------------------------------------------------#
	
		allowed_sequences = check_constituent_constraints(sequence_list, head_dictionary)

		if len(allowed_sequences) > 0:

			score = float(len(allowed_sequences)) / float(len(sequence_list))
			
			if score != current_score:
				return {tuple(current_move_list): score}
					
		return {tuple(current_move_list): 0.0}
	#---------------------------------------------------------------------#