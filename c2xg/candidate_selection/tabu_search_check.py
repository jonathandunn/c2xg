#-----------------------------------------------------------------------------#
def tabu_search_check(threshold_values, 
						n_checks, 
						full_vector_df, 
						all_indexes, 
						pos_unit_cost, 
						lex_unit_cost,
						cat_unit_cost,
						full_cxg, 
						Parameters, 
						run_parameter = 0
						):

	if run_parameter == 0:
		run_parameter = 1
	
		import multiprocessing as mp
		from functools import partial
		import random
		
		from candidate_selection.grammar_generator_random import grammar_generator_random
		from candidate_selection.get_grammar import get_grammar
		
		#First multi-process generation of n random grammars#
		print("\t\t\tGenerating random " + str(n_checks) + " grammars.")

		feature_list = [x for x in full_vector_df.columns if x[0:5] != "Index" and x != "Candidate" and x != "Encoded" and x != "Cost" and x != "Frequency"]
		ran_list = []
		for i in range(n_checks):
			ran_list.append(random.choice(feature_list))	

		pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = None)
		grammar_list = pool_instance.map(partial(grammar_generator_random, 
													threshold_values = threshold_values,
													), ran_list, chunksize = 200)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing#
			
		#Second multi-process evaluation of grammars#
		print("\t\t\tEvaluating generated grammars.")
		pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = None)
		eval_list = pool_instance.map(partial(get_grammar, 
												full_vector_df = full_vector_df, 
												all_indexes = all_indexes, 
												pos_unit_cost = pos_unit_cost, 
												lex_unit_cost = lex_unit_cost,
												cat_unit_cost = cat_unit_cost,
												full_cxg = full_cxg,
												save_grammar = False
												), grammar_list, chunksize = 1)
								
		pool_instance.close()
		pool_instance.join()
		#End multi-processing#
		
		#Find best grammar in results#
		result_dict = {}
		for i in range(len(eval_list)):
			result_dict[i] = eval_list[i]["mdl_full"]
			
		best_grammar = min(result_dict, key = result_dict.get)
		best_grammar = eval_list[best_grammar]
		
		return best_grammar
#-----------------------------------------------------------------------------#