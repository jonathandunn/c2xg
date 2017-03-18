#---------------------------------------------------------------------------------------------#
#INPUT: Phrase constituent list, lists of heads in both directions ---------------------------#
#OUTPUT: Updated direction-specific constituents organized by ngram length--------------------#
#---------------------------------------------------------------------------------------------#
def reformat_constituents(left_list, right_list):

	lr_constituent_dictionary = {}
	rl_constituent_dictionary = {}
	
	#Simplify from (Pos, Index) tuples to lists of indexes#
	left_list = [[x[1] for x in y] for y in left_list]
	right_list = [[x[1] for x in y] for y in right_list]
	
	#Begin loop through identified constituents#
	for rule in left_list:
	
		current_head = rule[0]
			
		try:
			lr_constituent_dictionary[current_head].append(rule)
							
		except:
			lr_constituent_dictionary[current_head] = []
			lr_constituent_dictionary[current_head].append(rule)
			
		
	for rule in right_list:
			
		current_head = rule[-1]
			
		try:
			rl_constituent_dictionary[current_head].append(rule)
								
		except:
			rl_constituent_dictionary[current_head] = []
			rl_constituent_dictionary[current_head].append(rule)
						
	return [lr_constituent_dictionary, rl_constituent_dictionary]
#---------------------------------------------------------------------------------------------#