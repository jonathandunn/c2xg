#-------------------------------------------------------------#
#-- Take grammar_dict, Return quality metric -----------------#
#-------------------------------------------------------------#
def grammar_score(grammar_dict, full_vector_df):

	from functions_candidate_pruning.get_grammar import get_grammar
	
	#Build query string from grammar dictionary#
	counter = 0
	first_flag = 1
	query_string = ""
	
	for feature_name in grammar_dict.keys():
		
		if grammar_dict[feature_name]["State"] == "On":
			counter += 1
			
			if counter == 20:
				query_string += ","
			
			if first_flag == 0:
				query_string += " | "
			
			query_string += "(" + feature_name + " > " + str(grammar_dict[feature_name]["Threshold"]) + ")"
			first_flag = 0

	metric_list = get_grammar(full_vector_df, query_string)
	
	try:
		metric = sum(metric_list) / float(len(metric_list))
		
	except:
		metric = 100

	return metric
#--------------------------------------------------------------#