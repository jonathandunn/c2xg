#-------------------------------------------------------------#
#-- Take grammar_dict, Return quality metric -----------------#
#-------------------------------------------------------------#
def format_optimum_grammar(grammar_dict, full_vector_df):

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

	#Limitation on number of conditions for query; ensure safety#
	double_flag = 0
	items = query_string.count(",")
	
	if items > 0:
		query_list = query_string.split(",")
		query_string = query_list[0]
		query_string2 = query_list[1]
		query_string2 = query_string2.replace(" | ","", 1).replace(",","")
		double_flag = 1	
	#End on safety check for overly long queries#
	
	training_list = [x for x in full_vector_df.columns.tolist() if x[0:8] == "Coverage"]
	
	if double_flag == 0:
		temp_df = full_vector_df.query(query_string, parser = "pandas", engine = "numexpr")
		
	elif double_flag == 1:
		temp_df = full_vector_df.query(query_string, parser = "pandas", engine = "numexpr")

		if len(temp_df) > 1 and "(" in query_string2:
			temp_df = temp_df.query(query_string2, parser = "pandas", engine = "numexpr")

	if len(temp_df) > 1:
			
		optimum_grammar = temp_df.loc[:"Candidates"].tolist()

	return optimum_grammar
#--------------------------------------------------------------#