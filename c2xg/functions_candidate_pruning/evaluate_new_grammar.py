#-----------------------------------------------------------------------------#
#--Evaluate possible changes to grammar --------------------------------------#
#-----------------------------------------------------------------------------#
def evaluate_new_grammar(updated_feature, grammar_dict, full_vector_df):

	from functions_candidate_pruning.grammar_score import grammar_score

	new_feature = updated_feature[0]
	new_threshold = updated_feature[1]
	
	new_dict = grammar_dict
	
	new_dict[new_feature]["Threshold"] = new_threshold
	new_dict[new_feature]["State"] = "On"
	
	new_score = grammar_score(new_dict, full_vector_df)
	
	return {new_score: new_dict}
#------------------------------------------------------------------------------#