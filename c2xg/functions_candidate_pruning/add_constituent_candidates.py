#---------------------------------------------------------------------------------------------#
#INPUT: Current candidates, phrase constituents, and expanded DataFrame filenames ------------#
#OUTPUT: Candidates with phrase constituents and their frequency added -----------------------#
#---------------------------------------------------------------------------------------------#
def add_constituent_candidates(full_candidate_list, phrase_constituent_list):
	
	for i in range(len(phrase_constituent_list)):
		
		current_dictionary = phrase_constituent_list[i]
		
		for key in current_dictionary.keys():
		
			current_list = current_dictionary[key]

			
			for sequence in current_list:
				
				sequence = eval(sequence)
			
				current_construction = []
				
				for unit in sequence:
					current_construction.append(('Pos', unit))
					
				current_label = str(current_construction)
				
				if current_label not in full_candidate_list:
					full_candidate_list.append(current_label)
					
	return full_candidate_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#