#--------------------------------------------------------------#
#--Take dictionary of initial feature weights -----------------#
#--Generate random combination of features with thresholds ----#
#--Return string for pandas query -----------------------------#
#--------------------------------------------------------------#
def grammar_generator(threshold_dict):
	
	from candidate_selection.reservoir_sampling import reservoir_sampling
	from random import randint
	
	feature_list = list(threshold_dict.keys())
	grammar_size = randint(1,len(feature_list))
	grammar_list = reservoir_sampling(feature_list, grammar_size)
	
	grammar_dict = {}
	
	for feature in grammar_list:
		
		type = randint(0,1)
		if type == 0:
			type = "AND"
		elif type == 1:
			type = "OR"
		
		grammar_dict[feature] = (threshold_dict[feature], type)

	return grammar_dict
#-------------------------------------------------------------#