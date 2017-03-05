#------------------------------------------------------------#
def get_grammar(grammar_dict,
					full_vector_df, 					 
					all_indexes, 
					pos_unit_cost, 
					lex_unit_cost,
					cat_unit_cost,
					full_cxg, 
					save_grammar = False
					):

	import pandas as pd

	from candidate_selection.grammar_query import grammar_query
	from candidate_selection.grammar_evaluator import grammar_evaluator
	
	query_string = grammar_query(grammar_dict)
	
	if len(full_vector_df) > 2:
	
		try:
			full_vector_df = full_vector_df.query(query_string, parser = "pandas", engine = "numexpr")
			
		except:
			full_vector_df = full_vector_df
		
	else:
		print("get_grammar.py line 19")
	
	#Get MDL metrics for current grammar#
	
	if len(full_vector_df) > 2:
	
		mdl_l1, mdl_l2, mdl_full = grammar_evaluator(full_vector_df.loc[:,["Candidate", "Encoded", "Indexes", "Cost"]], all_indexes)

	else:
		#print("Insufficient constructions in grammar to calculate MDL.")
		mdl_l1 = 100000000000000
		mdl_l2 = 100000000000000
		mdl_full = 1000000000000000000000
		
	grammar_dict["grammar_size"] = len(full_vector_df)
	grammar_dict["mdl_l1"] = mdl_l1
	grammar_dict["mdl_l2"] = mdl_l2
	grammar_dict["mdl_full"] = mdl_full
		
	if save_grammar == True:
		
		return grammar_dict, full_vector_df
		
	else:
	
		return grammar_dict
#-------------------------------------------------------------------#