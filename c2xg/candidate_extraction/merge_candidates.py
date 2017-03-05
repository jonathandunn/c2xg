#---------------------------------------------------------------------------------------------#
#INPUT: Template with numbers for column labels ----------------------------------------------#
#OUTPUT: List of template units with strings as column labels --------------------------------#
#---------------------------------------------------------------------------------------------#
def merge_candidates(data_file_candidate_constructions, sequence_list):
	
	from candidate_extraction.get_candidate_name import get_candidate_name
	from candidate_extraction.read_candidates import read_candidates
	
	#"Lemma" = 3, "Pos" = 4, "Category" = 5#
	
	full_candidate_list = []
	
	for original_template in sequence_list:
	
		template_name = []
		
		template = list(original_template)
		template = [x+3 for x in template]
	
		for i in range(len(template)):
			current_unit = template[i]
		
			template_name.append(current_unit)
					
		pickled_list_file = get_candidate_name(template_name, data_file_candidate_constructions)
		
		current_candidate_list = read_candidates(pickled_list_file)
		
		full_candidate_list.append(current_candidate_list)
	
	return full_candidate_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#