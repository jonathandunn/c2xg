#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def join_catenae_pairs(unit,
						pair_status_list, 
						head_status_dictionary,
						unit_list,
						input_files,
						pos_list,
						encoding_type,
						semantic_category_dictionary,
						word_list,
						lemma_list,
						lemma_dictionary,
						pos_dictionary,
						category_dictionary,
						delete_temp
						):
	
	import cytoolz as ct
	
	from functions_phrase_structure.join_right import join_right
	from functions_phrase_structure.join_left import join_left
	
	print("Starting " + str(unit))
	
	if unit in head_status_dictionary:
		
		current_direction = head_status_dictionary[unit]
				
		if current_direction == "L":
			current_rules = join_left(unit, 
										pair_status_list, 
										head_status_dictionary, 
										unit_list,
										input_files,
										encoding_type,
										semantic_category_dictionary,
										word_list,
										lemma_list,
										lemma_dictionary,
										pos_dictionary,
										category_dictionary,
										delete_temp
										)
				
		elif current_direction == "R":
			current_rules = join_right(unit, 
										pair_status_list, 
										head_status_dictionary, 
										unit_list,
										input_files,
										encoding_type,
										semantic_category_dictionary,
										word_list,
										lemma_list,
										lemma_dictionary,
										pos_dictionary,
										category_dictionary,
										delete_temp
										)
			
		
		return {unit: current_rules}
	
	else:
		return
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#