#---------------------------------------------------------------------------------------------#
#INPUT: Phrase constituent list, lists of heads in both directions ---------------------------#
#OUTPUT: Updated direction-specific constituents organized by ngram length--------------------#
#Take pos list and return updated pos list with phrase types added ---------------------------#
#---------------------------------------------------------------------------------------------#
def reformat_constituents(phrase_constituent_list, 
							lr_head_list, 
							rl_head_list, 
							max_ngram, 
							phrase_independence_list, 
							pos_list
							):

	lr_constituent_dictionary = {}
	rl_constituent_dictionary = {}
	
	#Begin loop through identified constituents#
	for i in range(len(phrase_constituent_list)):
		
		if phrase_constituent_list[i] != None:
		
			current_label = phrase_constituent_list[i][0]
			current_ngrams = phrase_constituent_list[i][1]
						
			if current_label in lr_head_list:
				lr_constituent_dictionary[current_label] = current_ngrams
				
			elif current_label in rl_head_list:
				rl_constituent_dictionary[current_label] = current_ngrams
		
	#Add independent heads as single-unit constituents#
	for head_tuple in phrase_independence_list:
		
		current_head = head_tuple[0]
		current_index = pos_list.index(current_head)
		current_status = head_tuple[1]
		
		if current_status == "Independent":
			if current_index in lr_head_list:
				temp_list = lr_constituent_dictionary[current_index]
				temp_list.append(str((current_index,)))
				lr_constituent_dictionary[current_index] = temp_list
				
			elif current_index in rl_head_list:
				temp_list = rl_constituent_dictionary[current_index]
				temp_list.append(str((current_index,)))
				rl_constituent_dictionary[current_index] = temp_list
				
	return [lr_constituent_dictionary, rl_constituent_dictionary]
#---------------------------------------------------------------------------------------------#