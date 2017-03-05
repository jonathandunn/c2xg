#--------------------------------------------------------------------------#
def tabu_search_restarts(mdl_file_list, test_file_list, Parameters, Grammar, max_candidate_length, full_cxg):

	import math
	import pandas as pd
	import time
	
	from candidate_extraction.read_candidates import read_candidates
	from candidate_extraction.write_candidates import write_candidates
	from candidate_selection.tabu_search_process import tabu_search_process
	from process_input.pandas_open import pandas_open
	from candidate_selection.grammar_evaluator import grammar_evaluator
	from candidate_selection.grammar_evaluator_baseline import grammar_evaluator_baseline
	from candidate_selection.construction_cost import construction_cost
	
	print("\t\tIs current grammar type a full CxG? " + str(full_cxg))
	print("")
	print("\t\tStarting Tabu Search ", end="")
	
	#Cycle through restarts, collecting output grammars#
	restart_list = []
	restart_counter = 0
	
	#PRE-CALCULATE AS MUCH OF MDL METRIC AS POSSIBLE#
	TOP_LEVEL_ENCODING = 0.301
	
	#Calculate base encoding costs#
	pos_units = len(Grammar.POS_List) 
	lex_units = len(Grammar.Lemma_List) 
	cat_units = len(Grammar.Category_List) 

	pos_unit_cost = -(math.log(1/float(pos_units))) + TOP_LEVEL_ENCODING
	lex_unit_cost = -(math.log(1/float(lex_units))) + TOP_LEVEL_ENCODING
	cat_unit_cost = -(math.log(1/float(cat_units))) + TOP_LEVEL_ENCODING
	
	#For full CxGs, there are three representation types to distinguish#
	if full_cxg == True:
		SLOT_R_COST = 0.4771
	
	#For other grammars, there is only one representation type without cost#
	else:
		SLOT_R_COST = 0.0000
		
		
	#Multi-process within each restart#
	for file in mdl_file_list:
	
		restart_counter += 1
		print("Restart " + str(restart_counter) + " on training file " + str(file))
		print("")
		
		#Load training_df and reduce to only measures used#
		training_df = read_candidates(file + ".MDL")
		
		#Drop unneeded features; only necessary because the query limit is 32#
		dropped_columns = ["Summed_RL_Unweighted", 
							"Summed_LR_Unweighted",
							"Summed_RL_Weighted",
							"Summed_LR_Weighted",
							"Directional_Scalar_Weighted",
							"Directional_Categorical_Weighted"
							]
		
		training_df.drop(dropped_columns, axis = 1, inplace = True)

		#Load restart file and pre-calculate for MDL metric#
		encoding_df = read_candidates(file + ".Training")
		num_units = encoding_df.loc[:,"Mas"].max()
		all_indexes = set(range(0,num_units+1))
		
		baseline_mdl = grammar_evaluator_baseline(encoding_df)
		
		del encoding_df
		
		#Calculate cost of each construction in advance#
		print("\t\tCalculating construction costs.")
		start_time = time.time()
		
		training_df = construction_cost(training_df, SLOT_R_COST, pos_unit_cost, lex_unit_cost, cat_unit_cost)
		
		print("\t\tTime to pre-calculate construction cost: " + str(time.time() - start_time))
		
		restart_list.append(tabu_search_process(training_df.copy("Deep"), 
												baseline_mdl, 
												all_indexes, 
												pos_unit_cost,
												lex_unit_cost,
												cat_unit_cost,
												full_cxg, 
												Parameters, 
												max_candidate_length
												))
												
	#Evaluate each output grammar on held-out test set#
	print("\t\tEvaluating restart grammars on held-out test set.")
	print("")
	
	#First, load testing files#
	testing_df = read_candidates(test_file_list[0] + ".MDL")
	encoding_df = read_candidates(test_file_list[0] + ".Training")
	
	num_units = encoding_df.loc[:,"Mas"].max()
	all_indexes = set(range(0,num_units+1))
	
	#Second, evaluate each restart against test set#
	mdl_dict = {}
	grammar_dict = {}
	
	for i in range(len(restart_list)):
		
		#Merge candidates from restart grammar with encoded data from training set#
		grammar_df = restart_list[i].loc[:,["Candidate", "State"]]
		restart_df = testing_df.merge(grammar_df, left_on = "Candidate", right_on = "Candidate")
		restart_df = restart_df.loc[:,["Candidate", "Encoded", "Indexes"]]
		restart_df = construction_cost(restart_df, SLOT_R_COST, pos_unit_cost, lex_unit_cost, cat_unit_cost)
				
		mdl_l1, mdl_l2, mdl_full = grammar_evaluator(restart_df.loc[:,["Candidate", "Encoded", "Indexes", "Cost"]], all_indexes)

		print("\t\tRestart " + str(i) + ": " + str(mdl_full))
		
		mdl_dict[i] = mdl_full
		grammar_dict[i] = restart_df.loc[:,"Candidate"].tolist()
		
	#Third, get best grammar and calculate stability metric for this fold#
	best_restart = min(mdl_dict, key = mdl_dict.get)
	best_grammar = set(grammar_dict[best_restart])
	best_mdl = mdl_dict[best_restart]
	
	print("")
	print("\t\tBest restart: " + str(best_restart))
	print("")

	print("\t\tCalculating stability metric.")
	stability_list = []
	
	for i in range(len(restart_list)):
	
		current_grammar = set(grammar_dict[i])
		current_mdl = mdl_dict[i]
		
		current_intersection = current_grammar.intersection(best_grammar)
		current_union = current_grammar | best_grammar
		current_agreement = len(current_intersection) / float(len(current_union))
		
		print("\t\t\tIntersection: " + str(len(current_intersection)) + "; Union: " + str(len(current_union)) + "; Agreement: " + str(current_agreement))
		
		absolute_difference = abs(best_mdl - current_mdl)
		relative_difference = absolute_difference / (float(best_mdl))
		adjusted_difference = 1 - relative_difference
		
		final_metric = current_agreement * adjusted_difference
		stability_list.append(final_metric)
		
		print("\t\t\tAbsolute difference: " + str(absolute_difference) + "; Relative difference: " + str(relative_difference) + "; Adjusted Difference: " + str(adjusted_difference))
		print("\t\t\t\tFinal metric: " + str(final_metric))
		print("")
		
	stability_metric = sum(stability_list) / float(len(stability_list))
	print("\t\tFinal stability metric: " + str(stability_metric))
	
	#Fourth, get merged grammar#
	print("")
	print("\t\tCreating merged grammar across restarts.")
	merged_grammar = best_grammar
	
	for key in grammar_dict:
	
		current_grammar = set(grammar_dict[key])
		merged_grammar = merged_grammar.union(current_grammar)
		
	print("\t\t\tSize of grammar merged across restarts: " + str(len(merged_grammar)))
	
	#Fifth, get final quality metric and weight by stability#	
	#Merge with training_df to get encoding data on the training set with best grammar candidates#
	merged_grammar_df = pd.Series(list(merged_grammar))
	merged_grammar_df = merged_grammar_df.to_frame(name = "Candidate")
	merged_grammar_df = merged_grammar_df.merge(testing_df, left_on = "Candidate", right_on = "Candidate")
	merged_grammar_df = construction_cost(merged_grammar_df, SLOT_R_COST, pos_unit_cost, lex_unit_cost, cat_unit_cost)
	
	merged_grammar_df = merged_grammar_df.loc[:,["Candidate", "Encoded", "Indexes", "Cost"]]
	
	#Get full MDL for comparison against unencoded baseline#
	mdl_l1, mdl_l2, mdl_full = grammar_evaluator(merged_grammar_df, all_indexes)
															
	#Calculate unencoded baseline#
	#Reload encoding_df because it was reduced to only zero indexes above#
	encoding_df = read_candidates(test_file_list[0] + ".Training")
	total_unencoded_size = grammar_evaluator_baseline(encoding_df)
	total_over_baseline = 1 - (mdl_full / (float(total_unencoded_size)))
	
	print("")
	print("\t\tTest MDL (Full): " + str(mdl_full) + "; Unencoded MDL: " + str(total_unencoded_size) + "; Adjusted MDL: " + str(total_over_baseline))
	
	weighted_mdl = total_over_baseline * stability_metric
	
	print("\t\tFinal stability weighted metric: " + str(weighted_mdl))

	return merged_grammar_df.loc[:,"Candidate"], weighted_mdl
#--------------------------------------------------------------------------#