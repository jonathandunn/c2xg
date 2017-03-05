#--------------------------------------------------------------#
#--Take dictionary of possible feature weights ----------------#
#--Return random grammar --------------------------------------#
#--------------------------------------------------------------#
def grammar_generator_random(current_feature, threshold_values, random_state = True, fixed_size = 0):
	
	from candidate_selection.reservoir_sampling import reservoir_sampling
	import random

	feature_list = list(threshold_values.keys())
	
	if fixed_size == 0:
		grammar_size = random.randint(1,len(feature_list))
	else:
		grammar_size = fixed_size
			
	grammar_list = reservoir_sampling(feature_list, grammar_size)
		
	if current_feature not in grammar_list:
		grammar_list.append(current_feature)
		
	grammar_dict = {}
		
	for feature in grammar_list:

		if random_state == True:
			type = random.randint(0,1)
			if type == 0:
				type = "AND"
			elif type == 1:
				type = "OR"
			
		else:
			type = "AND"
			
		temp_parameter_list = list(threshold_values[feature])
		temp_parameter_value = random.choice(temp_parameter_list)
		grammar_dict[feature] = (temp_parameter_value, type)
			
	return grammar_dict
#-------------------------------------------------------------#