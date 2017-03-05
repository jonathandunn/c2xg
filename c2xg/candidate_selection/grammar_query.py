#--------------------------------------------------------------#
def grammar_query(grammar_dict):
	
	and_counter = 0
	or_counter = 0
	
	and_query = "("
	or_query = "("
	
	for feature in grammar_dict.keys():
	
		if feature not in ["Candidate", "Frequency", "Encoded", "Indexes", "mdl_l1", "mdl_l2", "mdl_l3", "mdl_full", "grammar_size"]:
			
			current_threshold = str(grammar_dict[feature][0])
			current_type = str(grammar_dict[feature][1])

			if current_type == "AND":
					
					and_counter += 1
					
					if and_counter > 1:
						and_query += " & "
						
					and_query += str(feature) + " > " + str(current_threshold)
									
			elif current_type == "OR":
					
					or_counter += 1
					
					if or_counter > 1:
						or_query += " | "
						
					or_query += "(" + str(feature) + " > " + str(current_threshold) + ")"
					
	query_string = ""
	
	if and_counter >= 1:
		and_query += ")"
		query_string += and_query
		
	if or_counter >= 1:
		or_query += ")"
		
		if and_counter >= 1:
			query_string += " | " + or_query
		
		else:
			query_string += or_query
	
	return query_string
#-------------------------------------------------------------#