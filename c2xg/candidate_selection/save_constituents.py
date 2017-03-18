#---------------------------------------------------------------------#
def save_constituents(final_grammar, Parameters, Grammar, run_parameter = 0):

	from candidate_selection.reformat_constituents import reformat_constituents
	from candidate_selection.write_grammar_debug import write_grammar_debug
	from candidate_extraction.write_candidates import write_candidates
	
	# NON-HEAD:
    
		# adj: adjective
		# adv: adverb	
		# aux: auxiliary verb	
		# intj: interjection
		# part: particle	
		# det: determiner
		# num: numeral
		# conj: coordinating conjunction
	
	# HEAD:
    
		# pron: pronoun
		# noun: noun
		# propn: proper noun
		# adp: adposition
		# verb: verb
		# sconj: subordinating conjunction
	
	print("Assigning heads to constituents")
	
	head_list = ["pron", "noun", "propn", "adp", "verb", "sconj"]
	non_head_list = ["adj", "adv", "aux", "intj", "part", "det", "num", "conj"]
	
	head_indexes = [Grammar.POS_List.index(x) for x in head_list if x in Grammar.POS_List]
	non_head_indexes = [Grammar.POS_List.index(x) for x in non_head_list if x in Grammar.POS_List]
	
	right_list = []
	left_list = []
	
	no_good_counter = 0
	full_counter = 0
	
	#Go through all sequences#
	for sequence in final_grammar:
	
		full_counter += 1
		
		if sequence[0][1] in non_head_indexes and sequence[-1][1] in non_head_indexes:
			#print("Not allowed: " + str(sequence))
			no_good_counter += 1
			
		elif sequence[0][1] in head_indexes and sequence[-1][1] in non_head_indexes:
			#print("Left-headed: " + str(sequence))
			left_list.append(sequence)
		
		elif sequence[0][1] in non_head_indexes and sequence[-1][1] in head_indexes:
			#print("Right-headed: " + str(sequence))
			right_list.append(sequence)
			
		elif sequence[0][1] in head_indexes and sequence[-1][1] in head_indexes:
			#print("Both end-points are heads!")
			left = head_indexes.index(sequence[0][1])
			right = head_indexes.index(sequence[-1][1])
			
			if left > right:
				#print("Left-headed: " + str(sequence))
				left_list.append(sequence)
				
			elif right > left:
				#print("Right-headed: " + str(sequence))	
				right_list.append(sequence)
				
	print("Done assigning heads: " + str(no_good_counter) + " removed out of " + str(full_counter) + " total.")
	
	write_grammar_debug(left_list, "Left-Heads." + "Constituents", Grammar, Parameters)
	write_grammar_debug(right_list, "Right-Heads." + "Constituents", Grammar, Parameters)
			
	#Now format and save constituent grammar#
	Grammar.Constituent_Dict = reformat_constituents(left_list, right_list)
	Grammar.Type = "Constituent"
	
	write_candidates(Parameters.Data_File_Constituents, Grammar)
			
	return Grammar
#------------------------------------------------------------------------------#