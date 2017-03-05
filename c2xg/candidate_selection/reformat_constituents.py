#---------------------------------------------------------------------------------------------#
#INPUT: Phrase constituent list, lists of heads in both directions ---------------------------#
#OUTPUT: Updated direction-specific constituents organized by ngram length--------------------#
#---------------------------------------------------------------------------------------------#
def reformat_constituents(sequence_list, optimum_head_dictionary):

	from candidate_selection.check_constituent_constraints import check_constituent_constraints
	
	allowed_sequences = check_constituent_constraints(sequence_list, optimum_head_dictionary)

	lr_constituent_dictionary = {}
	rl_constituent_dictionary = {}
	
	#Begin loop through identified constituents#
	for rule in allowed_sequences:
	
		if optimum_head_dictionary[rule[0]]["Direction"] == 1:
			
			current_head = rule[0]
			
			try:
				lr_constituent_dictionary[current_head].append(rule)
							
			except:
				lr_constituent_dictionary[current_head] = []
				lr_constituent_dictionary[current_head].append(rule)
			
		elif optimum_head_dictionary[rule[-1]]["Direction"] == -1:
			
			current_head = rule[-1]
			
			try:
				rl_constituent_dictionary[current_head].append(rule)
								
			except:
				rl_constituent_dictionary[current_head] = []
				rl_constituent_dictionary[current_head].append(rule)
						
	return [lr_constituent_dictionary, rl_constituent_dictionary]
#---------------------------------------------------------------------------------------------#