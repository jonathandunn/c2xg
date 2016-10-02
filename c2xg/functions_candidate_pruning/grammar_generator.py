#--------------------------------------------------------------#
#--Take dictionary of initial feature weights -----------------#
#--Generate random combination of features with thresholds ----#
#--Return string for pandas query -----------------------------#
#--------------------------------------------------------------#
def grammar_generator(threshold_dict):
	
	from functions_candidate_pruning.reservoir_sampling import reservoir_sampling
	from random import randint
	
	feature_list = list(threshold_dict.keys())
	grammar_size = randint(1,len(feature_list))
	
	grammar_list = reservoir_sampling(feature_list, grammar_size)
	
	query_string = ""
	first_flag = 1
	counter = 0
	
	for feature in grammar_list:
	
		counter += 1
		current_threshold = threshold_dict[feature]
		
		if first_flag == 0:
			query_string += " | "
			
		query_string += "(" + feature + " > " + str(current_threshold) + ")"
		first_flag = 0
		
		if counter > 30:
			query_string += ","

	return query_string
#-------------------------------------------------------------#