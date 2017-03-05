#-----------------------------------------------------------------------------#
def move_evaluator(current_move, 
					grammar_dict, 
					full_vector_df, 
					all_indexes, 
					pos_unit_cost, 
					lex_unit_cost,
					cat_unit_cost,
					full_cxg,
					current_best
					):

	from candidate_selection.get_grammar import get_grammar
	return_tuple = []
	
	for current_parameter in current_move:
		grammar_dict[current_parameter] = current_move[current_parameter]
		return_tuple.append((current_parameter, current_move[current_parameter]))
		
	#Update grammar with current move#
	grammar_dict["mdl_full"] = 1000000000000000000000

	#Get MDL metric back#
	grammar_dict = get_grammar(grammar_dict,
								full_vector_df, 								 
								all_indexes, 
								pos_unit_cost,
								lex_unit_cost,
								cat_unit_cost,
								full_cxg, 
								save_grammar = False
								)
								
	if grammar_dict["mdl_full"] != 1000000000000000000000 and grammar_dict["mdl_full"] != current_best:	
								
		return_tuple = tuple(return_tuple)
		return {return_tuple: grammar_dict["mdl_full"]}
		
	else:
		return {}
#-----------------------------------------------------------------------------#