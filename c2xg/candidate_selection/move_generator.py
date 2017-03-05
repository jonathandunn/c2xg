#-----------------------------------------------------------------------------#
def move_generator(current_feature, grammar_dict, threshold_values, checks_per_move, max_move_size):

	import random
	import statistics
	
	from candidate_selection.grammar_generator_random import grammar_generator_random
	
	return_list = []
	move_list = []
	
	#FIRST, GET OR CANDIDATES, HALF UP FROM CURRENT AND HALF DOWN#
	checks_per_move_or = 2
	checks_per_move = checks_per_move - (checks_per_move_or * 2)
	
	#If feature is currently on, randomly find thresholds above and below current#
	if current_feature in grammar_dict:
		
		current_threshold = grammar_dict[current_feature][0]
		current_threshold_values = threshold_values[current_feature]
			
	#IF a feature has no current threshold, find the median value#
	else:
		current_threshold_values = threshold_values[current_feature]
		current_threshold = statistics.median(current_threshold_values)
		
	#Sometimes a parameter has fewer settings than available moves#
	if checks_per_move_or > len(current_threshold_values):
		move_list = current_threshold_values
		
	#Usually not#
	else:
		
		#Get as many values above / below threshold as possible#
		above_values = [x for x in current_threshold_values if x > current_threshold]
		below_values = [x for x in current_threshold_values if x < current_threshold]
			
		if len(above_values) < checks_per_move_or:
			move_list += above_values
		
		else:
			move_list += random.sample(above_values, checks_per_move_or)
					
		if len(below_values) < checks_per_move_or:
			move_list += below_values
					
		else:
			move_list += random.sample(below_values, checks_per_move_or)
			
	#Add OR moves to list#			
	for move in move_list:
		return_list.append({current_feature: (move, "OR")})
		
	#Add an "OFF" move for current feature#
	return_list.append({current_feature: (0, "OFF")})
	
	#SECOND, GET RANDOM MOVES CONTAINING THE CURRENT FEATURE AMONG OTHERS#
	for i in range(checks_per_move):
		
		fixed_size = type = random.randint(2,max_move_size)
		move_dict = grammar_generator_random(current_feature, threshold_values, random_state = False, fixed_size = fixed_size)
		return_list.append(move_dict)

	return return_list
#-----------------------------------------------------------------------------#