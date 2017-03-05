#-----------------------------------------------------------------------------#
def move_evaluator_direct(current_index, 
					move_list,
					full_vector_df, 
					all_indexes, 
					pos_unit_cost, 
					lex_unit_cost,
					cat_unit_cost,
					full_cxg,
					current_best
					):

	from candidate_selection.grammar_evaluator import grammar_evaluator

	#Get current reference move#
	current_move = move_list[current_index]
	
	#Reverse indexes in current move by multiplying by -1 #
	full_vector_df.loc[current_move, "State"] = full_vector_df.loc[current_move, "State"] * -1
	full_vector_df = full_vector_df[full_vector_df.State != -1]

	#Evaluate for MDL metric#
	mdl_l1, mdl_l2, mdl_full = grammar_evaluator(full_vector_df.loc[:,["Candidate", "Encoded", "Indexes", "Cost"]], all_indexes)
								
	if mdl_full != 1000000000000000000000 and mdl_full != current_best:	
								
		return {current_index: mdl_full}
		
	else:
		return {}
#-----------------------------------------------------------------------------#