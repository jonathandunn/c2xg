#--------------------------------------------------------------#
#--Processing function to search through potential grammars, --#
#-- evaluate each, and return optimum grammar found------------#
#--------------------------------------------------------------#
def find_optimum_grammar(full_vector_df, 
						number_of_cpus, 
						run_parameter = 0
						):

	if run_parameter == 0:
		run_parameter = 1
		
		import pandas as pd
		import numpy as np
		import cytoolz as ct
		import multiprocessing as mp
		from functools import partial
		import time
		
		from functions_candidate_pruning.learn_thresholds import learn_thresholds
		from functions_candidate_pruning.learn_feature_set import learn_feature_set
		from functions_candidate_pruning.grammar_score import grammar_score
		from functions_candidate_pruning.get_grammar_changes import get_grammar_changes
		from functions_candidate_pruning.evaluate_new_grammar import evaluate_new_grammar
		from functions_candidate_pruning.format_optimum_grammar import format_optimum_grammar
		
		from functions_candidate_extraction.write_candidates import write_candidates
		from functions_candidate_extraction.read_candidates import read_candidates
		
		#First, multi-process the search for optimum threshold values#
		starting = time.time()
		print("Learning initial feature weights.")
		
		feature_list = [x for x in full_vector_df.columns if x[0:8] != "Coverage" and x != "Candidate"]
		
		pool_instance=mp.Pool(processes = number_of_cpus, maxtasksperchild = None)
		threshold_list = pool_instance.map(partial(learn_thresholds, 
												full_vector_df = full_vector_df.copy("Deep")
												), feature_list, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing#
		
		threshold_dict = ct.merge([x for x in threshold_list])
		del threshold_list
		
		print("")
		print("Total time for learning threshold values: " + str(time.time() - starting))
		print("")
		
		#Second, multi-process the search for optimum set of features#
		starting = time.time()
		print("Learning optimum set of features.")
		
		pool_instance=mp.Pool(processes = number_of_cpus, maxtasksperchild = None)
		state_list = pool_instance.map(partial(learn_feature_set, 
												full_vector_df = full_vector_df.copy("Deep"),
												threshold_dict = threshold_dict
												), feature_list, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing#
		
		print("")
		print("Total time for learning optimum set of features: " + str(time.time() - starting))
		print("")	
		
		state_dict = ct.merge([x for x in state_list])
		del state_list
		
		#Merge state and threshold dictionaries into grammar dictionary#
		grammar_dict = {}
		for feature_name in feature_list:
			grammar_dict[feature_name] = {}
			grammar_dict[feature_name]["Threshold"] = threshold_dict[feature_name]
			grammar_dict[feature_name]["State"] = state_dict[feature_name]
			
		starting_score = grammar_score(grammar_dict, full_vector_df)
		
		#Third, alter features individually, keeping improvements, until stable state reached#
		no_change_counter = 0
		total_loop_counter = 0
		
		#Evaluate current grammar to get baseline score#
		current_score = grammar_score(grammar_dict, full_vector_df)
		
		while True:
		
			total_loop_counter += 1
			print("Adjust feature weights, iteration " + str(total_loop_counter))
			
			#Loop through each feature to evaluate small changes in that feature#
			for feature_name in feature_list:
			
				#Multi-process 100 changes to the current feature#
				change_list = get_grammar_changes(feature_name, grammar_dict, full_vector_df)
								
				#Start multi-processing feature changes#
				pool_instance=mp.Pool(processes = number_of_cpus, maxtasksperchild = None)
				result_list = pool_instance.map(partial(evaluate_new_grammar, 
														full_vector_df = full_vector_df,
														grammar_dict = grammar_dict
														), change_list, chunksize = 1)
				pool_instance.close()
				pool_instance.join()
				#End multi-processing feature changes#
				
				#Merge results and find best score and grammar#
				result_dict = ct.merge([x for x in result_list])
				best_score = min(list(result_dict.keys()))
				best_grammar = result_dict[best_score]
				print(current_score, end="")
				print(" and ", end="")
				print(best_score)
				#Choose the best change if better than baseline#
				if float(best_score) < float(current_score):
					no_change_counter = 0
					current_score = best_score
					grammar_dict = best_grammar
					
					grammar_dict[feature_name]["State"] = "On"
					
					print("\tNew best grammar score " + str(current_score) + ", from changing " + feature_name)
					
				else:
					no_change_counter += 1				
			
			#After loop through features, check if a stable state has been reached#
			if no_change_counter == 5 or total_loop_counter > 1000000:
				break
		
		#End while loop#
		print("Score before adjusting feature weights: " + str(starting_score))
		print("Score after adjusting feature weights: " + str(current_score))
		
		optimum_grammar = format_optimum_grammar(grammar_dict, full_vector_df)
		print(optimum_grammar)

		return optimum_grammar
#-------------------------------------------------------------#