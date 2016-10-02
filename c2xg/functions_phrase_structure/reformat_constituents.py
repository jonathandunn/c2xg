#---------------------------------------------------------------------------------------------#
#INPUT: Phrase constituent list, lists of heads in both directions ---------------------------#
#OUTPUT: Updated direction-specific constituents organized by ngram length--------------------#
#Take pos list and return updated pos list with phrase types added ---------------------------#
#---------------------------------------------------------------------------------------------#
def reformat_constituents(cfg_dictionary):

	lr_constituent_dictionary = {}
	rl_constituent_dictionary = {}
	
	#Begin loop through identified constituents#
	for rule in cfg_dictionary.keys():
	
		if cfg_dictionary[rule] == "L":
			current_head = rule[0]
			
			try:
				lr_constituent_dictionary[current_head].append(rule)
							
			except:
				lr_constituent_dictionary[current_head] = []
				lr_constituent_dictionary[current_head].append(rule)
			
		elif cfg_dictionary[rule] == "R":
			current_head = rule[-1]
			
			try:
				rl_constituent_dictionary[current_head].append(rule)
								
			except:
				rl_constituent_dictionary[current_head] = []
				rl_constituent_dictionary[current_head].append(rule)
						
	return [lr_constituent_dictionary, rl_constituent_dictionary]
#---------------------------------------------------------------------------------------------#