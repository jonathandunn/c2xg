#--------------------------------------------------------------#
#--Processing function to search through potential grammars, --#
#-- evaluate each, and return optimum grammar found------------#
#--------------------------------------------------------------#
def tabu_search_process(full_vector_df, 
						baseline_mdl, 
						all_indexes, 
						pos_unit_cost, 
						lex_unit_cost,
						cat_unit_cost,
						full_cxg, 
						Parameters,
						max_candidate_length, 
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
		from collections import deque
		
		from candidate_selection.get_threshold_ranges import get_threshold_ranges
		from candidate_selection.get_grammar import get_grammar
		from candidate_selection.move_generator import move_generator
		from candidate_selection.move_evaluator import move_evaluator
		from candidate_selection.move_generator_direct import move_generator_direct
		from candidate_selection.move_evaluator_direct import move_evaluator_direct
		from candidate_selection.tabu_search_check import tabu_search_check
		
		from candidate_extraction.write_candidates import write_candidates
		from candidate_extraction.read_candidates import read_candidates
		
		#Initialize list of features and list of thresholds#
		feature_list = [x for x in full_vector_df.columns if x[0:5] != "Index" and x != "Candidate" and x != "Encoded" and x != "Frequency"]
		threshold_values = get_threshold_ranges(feature_list, full_vector_df, Parameters.Tabu_Thresholds_Number, max_candidate_length)
		
		print("\t\t\tPruning unobserved candidates.")
		full_vector_df = full_vector_df[full_vector_df.loc[:,"Encoded"] != 0]
		print("\t\t\tNumber of candidates after pruning unobserved: " + str(len(full_vector_df)))
		print("")
		
		#Randomly initialization grammar sampler#
		print("\t\t\tRandomly initializing the Association-Based Grammar Sampler for " + str(len(full_vector_df)) + " candidates.")
		starting = time.time()

		best_grammar_dict = tabu_search_check(threshold_values, 
												Parameters.Tabu_Random_Checks, 
												full_vector_df.copy("Deep"), 
												all_indexes, 
												pos_unit_cost,
												lex_unit_cost,
												cat_unit_cost,
												full_cxg, 
												Parameters
												)
				
		print("")
		print("\t\t\tInitial grammar state:")
		
		for feature in feature_list:
			if feature in best_grammar_dict:
				print("\t\t\t" + str(feature) + ": " + str(best_grammar_dict[feature]))
		
		print("")
		print("\t\t\tBest initial starting state: " + str(best_grammar_dict["mdl_full"]) + " against baseline of " + str(baseline_mdl))
		print("\t\t\tTime to evaluate random grammars: " + str(time.time() - starting))
		
		#Initialize main tabu search loop#
		no_change_counter = 0
		total_loop_counter = 0
		tabu_list = deque([], maxlen = 7)
		
		#Evaluate current grammar to get baseline score#
		best_grammar_dict = get_grammar(best_grammar_dict,
											full_vector_df.copy("Deep"), 											 
											all_indexes, 
											pos_unit_cost, 
											lex_unit_cost,
											cat_unit_cost,
											full_cxg
											)
		
		starting_score = best_grammar_dict["mdl_full"]
		current_grammar_dict = best_grammar_dict.copy()
		
		#----START TABU SEARCH LOOP --------------------------------------------------------------------------------------#		
		while True:
			
			total_loop_counter += 1
			print("")
			print("")
			print("\t\tStarting tabu search loop number " + str(total_loop_counter) + " with " + str(no_change_counter) + " loops since new best state found and " + str(len(full_vector_df)) + " total candidates.")
			print("\t\tCurrent tabu list: ", end="")
			print(tabu_list)
			print("\t\tCurrent best grammar MDL: " + str(best_grammar_dict["mdl_full"]) + " with " + str(best_grammar_dict["grammar_size"]) + " constructions.")
			
			if total_loop_counter > 1:
				print("\t\t\tTime for previous loop: " + str(time.time() - loop_start))
			loop_start = time.time()

			#----------------------------------------------------------------------------------------------------#
			#Multi-process Generate possible moves, formatted as {"Threshold": New_Parameter} -------------------#
			starting = time.time()
			
			print("\t\t\tMulti-processing move generation.")
			pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = None)
			move_list = pool_instance.map(partial(move_generator, 
												grammar_dict = current_grammar_dict.copy(),
												threshold_values = threshold_values,
												checks_per_move = Parameters.Tabu_Indirect_Move_Number,
												max_move_size = Parameters.Tabu_Indirect_Move_Size
												), feature_list, chunksize = 25)
			pool_instance.close()
			pool_instance.join()
						
			move_list = [item for sublist in move_list for item in sublist]

			#----End multi-processing for move generation-------------------------------------------------------#
			
			#---------------------------------------------------------------------------------------------------#
			#Multi-process Evaluation of possible moves---------------------------------------------------------#
			print("\t\t\tMulti-processing move evaluation.")
			pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = None)
			move_eval_list = pool_instance.map(partial(move_evaluator, 
												grammar_dict = current_grammar_dict.copy(),
												full_vector_df = full_vector_df.copy("Deep"), 
												all_indexes = all_indexes, 
												pos_unit_cost = pos_unit_cost, 
												lex_unit_cost = lex_unit_cost,
												cat_unit_cost = cat_unit_cost,
												full_cxg = full_cxg,
												current_best = best_grammar_dict["mdl_full"]
												), move_list, chunksize = 25)
			pool_instance.close()
			pool_instance.join()

			move_eval_dict = ct.merge(move_eval_list)
			del move_eval_list
			
			print("\t\t\tTime for generating and evaluating moves for current turn: " + str(time.time() - starting))
			#----End multi-processing for move evaluation -----------------------------------------------------#
			
			#-----------LOOP UNTIL BEST ALLOWED MOVE IS FOUND--------------------------------------------------#
			starting = time.time()
			while True:
				print("\t\t\tNow determining best available move.")

				#Check to ensure moves have not been exhausted#
				if len(move_eval_dict) > 0:
					
					best_move = min(move_eval_dict, key = move_eval_dict.get)
					best_move_parameters = [x[0] for x in best_move]
					
					best_move_type = best_move[0][1][1]
					best_move_mdl = move_eval_dict[best_move]

					#Check if best move is allowed by tabu list#
					print("\t\t\t\tCandidate best move is " + str(best_move))
					
					#Flatten tabu list and check if tabu is violated#
					if len(tabu_list) > 0:
						flat_tabu_list = [item for sublist in tabu_list for item in sublist]
					else:
						flat_tabu_list = []
					
					tabu_flag = False
					for one_parameter in best_move_parameters:
						if one_parameter in flat_tabu_list:
							tabu_flag = True
					#Done checking if tabu is violated#
					
					if tabu_flag == True or best_move_type == "OR":
						print("\t\t\t\tBest move is in tabu list. Checking aspiration criteria.")
					
						#Check if MDL score overrules tabu list#
						if best_move_mdl < best_grammar_dict["mdl_full"]:
							print("\t\t\t\tBest move is allowed: tabu overruled.")
							no_best_move_flag = False
							break
							
						else:
							#print("\t\t\t\tBest move is not allowed. Moving to next best move.")
							move_eval_dict.pop(best_move)
												
					#If best move not in tabu list, stop searching#
					else:
						no_best_move_flag = False
						break
						
				#If moves have been exhausted, do nothing#
				else:
					print("\t\t\t\tNo candidates in current turn are best allowed. Starting next turn without changing state.")
					no_best_move_flag = True
					break
			
			#-----------END LOOP FOR FINDING BEST MOVE -------------------------------------------------------#
						
			if no_best_move_flag == False:
			
				#Make move and update Tabu list#
				print("")
				print("\t\t\tCurrent best move: " + str(best_move) + " with MDL of " + str(best_move_mdl))
				tabu_list.appendleft(best_move_parameters)
				print(tabu_list)
				
				#Update current_grammar_dict#
				for move in best_move:
					current_parameter = move[0]
					current_state = move[1]
					
					current_grammar_dict[current_parameter] = current_state
				
				#Evaluate current grammar#
				current_grammar_dict = get_grammar(current_grammar_dict, 
														full_vector_df.copy("Deep"), 													
														all_indexes, 
														pos_unit_cost, 
														lex_unit_cost,
														cat_unit_cost,
														full_cxg
														)
				
				if current_grammar_dict["mdl_full"] < best_grammar_dict["mdl_full"]:
					
					print("\t\t\tNew best grammar: ", end = "")
					print(best_move)
					
					no_change_counter = 0
					best_grammar_dict = current_grammar_dict.copy()
					print("\t\t\tTime to check move status and make best move: " + str(time.time() - starting))	
					
				else:
					no_change_counter += 1
					print("\t\t\tMove does not reach new global optimum.")
					print("\t\t\tTime to check move status and make best move: " + str(time.time() - starting))	
			
			#If no best move to evaluate, count turn as a loss#
			elif no_best_move_flag == True:
				no_change_counter += 1
				print("\t\t\tTime to check move status and make best move: " + str(time.time() - starting))	

			#Check for stopping condition#
			if no_change_counter >= 14:
				print("")
				print("")
				print("\t\tNo best grammar found for two cycles through tabu list. Performing validation safety check.")
				print("")
				
				#Get the score for n random states and check if any provides a better grammar#
				starting = time.time()

				best_check_grammar_dict = tabu_search_check(threshold_values, 
															Parameters.Tabu_Random_Checks, 
															full_vector_df.copy("Deep"), 
															all_indexes, 
															pos_unit_cost, 
															lex_unit_cost,
															cat_unit_cost,
															full_cxg, 
															Parameters
															)
				
				print("t\t\tTime to evaluate random grammars: " + str(time.time() - starting))
				
				if best_check_grammar_dict["mdl_full"] < best_grammar_dict["mdl_full"]:
				
					print("")
					print("\t\tSURPRISE: RANDOM STATE EXCEEDS BEST FOUND GRAMMAR. RESTARTING FROM THIS STATE.")
					print("")
					
					best_grammar_dict = best_check_grammar_dict.copy()
					current_grammar_dict = best_check_grammar_dict.copy()
					
				else:
					
					print("\t\tRandom validation states do not exceed optimum grammar. Stopping search.")
					
					break
					
		#----END TABU SEARCH LOOP --------------------------------------------------------------------------------------#		
			
		print("")
		print("\tScore before indirect tabu search: " + str(starting_score))
		print("\tScore after indirect tabu search: " + str(best_grammar_dict["mdl_full"]))
		print("\tBaseline Score: " + str(baseline_mdl))
		print("")
		
		print("")
		print("\tBeginning direct tabu search.")
		print("")
		
		starting = time.time()
		starting_direct_score = best_grammar_dict["mdl_full"]
		
		#First, reduce full_vector_df to constructions, state, and coverage values#
		best_grammar_dict, tabu_search_df = get_grammar(best_grammar_dict, 
														full_vector_df, 													
														all_indexes, 
														pos_unit_cost, 
														lex_unit_cost,
														cat_unit_cost,
														full_cxg,
														save_grammar = True
														)
		
		tabu_search_df = tabu_search_df.loc[:,["Candidate", "Encoded", "Indexes"]]
		tabu_search_df.loc[:,"State"] = 1
				
		#Second, randomly initialize state by evaluating randomly generate grammars#
		print("\t\tInitializing direct tabu search.")
		
		move_list = move_generator_direct(tabu_search_df, Parameters.Tabu_Random_Checks, Parameters.Tabu_Direct_Move_Size)
		move_index = [x for x in range(0,Parameters.Tabu_Random_Checks-1)]
		
		pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = None)
		move_eval_list = pool_instance.map(partial(move_evaluator_direct, 
												move_list = move_list,
												full_vector_df = tabu_search_df.copy("Deep"), 
												all_indexes = all_indexes, 
												pos_unit_cost = pos_unit_cost, 
												lex_unit_cost = lex_unit_cost,
												cat_unit_cost = cat_unit_cost,
												full_cxg = full_cxg,
												current_best = best_grammar_dict["mdl_full"]
												), move_index, chunksize = 50)
		pool_instance.close()
		pool_instance.join()
			
		move_eval_dict = ct.merge(move_eval_list)
		best_move = min(move_eval_dict, key = move_eval_dict.get)
		current_best_mdl = move_eval_dict[best_move]
		best_move = move_list[best_move]	

		tabu_search_df.loc[best_move, "State"] = tabu_search_df.loc[best_move, "State"] * -1
		tabu_search_mdl = current_best_mdl
		optimum_grammar_df = tabu_search_df.copy("Deep")
		optimum_grammar_mdl = current_best_mdl
		starting_score_direct = optimum_grammar_mdl
		
		#Third, Tabu search search across randomly generate moves#
		print("\t\tStarting direct tabu search loop.")
		no_change_counter = 0
		total_loop_counter = 0
		tabu_list = deque([], maxlen = 14)
		
		while True:
		
			total_loop_counter += 1
			
			print("")
			print("")
			print("\t\t\tStarting loop number " + str(total_loop_counter) + " with " + str(no_change_counter) + " loops since last improvement.")
			print("\t\t\tTabu list: ", end="")
			print(tabu_list)
			
			if total_loop_counter > 1:
				print("\t\t\tTime for previous loop: " + str(time.time() - loop_start))
			loop_start = time.time()
			
			#Generate and evaluate potential moves#
			move_list = move_generator_direct(tabu_search_df, Parameters.Tabu_Direct_Move_Number, Parameters.Tabu_Direct_Move_Size)
			move_index = [x for x in range(0,Parameters.Tabu_Direct_Move_Number-1)]
		
			pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = None)
			move_eval_list = pool_instance.map(partial(move_evaluator_direct, 
													move_list = move_list,
													full_vector_df = tabu_search_df.copy("Deep"), 
													all_indexes = all_indexes, 
													pos_unit_cost = pos_unit_cost, 
													lex_unit_cost = lex_unit_cost,
													cat_unit_cost = cat_unit_cost,
													full_cxg = full_cxg,
													current_best = tabu_search_mdl
													), move_index, chunksize = 50)
			pool_instance.close()
			pool_instance.join()
				
			move_eval_dict = ct.merge(move_eval_list)
			
			#WHILE loop for choosing best move, checking tabu status and aspiration criteria#
			while True:
			
				#Make sure there are current moves available#
				if len(list(move_eval_dict.keys())) >= 1:
								
					best_move = min(move_eval_dict, key = move_eval_dict.get)
					best_move_mdl = move_eval_dict[best_move]
					best_move_index = best_move
					best_move = move_list[best_move]

					print("\t\t\tCurrent best move: " + str(best_move))
					
					tabu_flag = False
					flat_tabu_list = [item for sublist in tabu_list for item in sublist]
				
					for i in best_move:
						if i in flat_tabu_list:
							tabu_flag = True
							
					#If best move is tabu#
					if tabu_flag == True:
					
						print("\t\t\tBest move is on tabu list. Checking aspiration criteria.")
						
						if best_move_mdl < optimum_grammar_mdl:
						
							print("\t\t\tAspiration criteria satisfied: New Best Grammar with MDL metric of " + str(best_move_mdl) + " against baseline of " + str(baseline_mdl))
							tabu_search_df.loc[best_move, "State"] = tabu_search_df.loc[best_move, "State"] * -1
							optimum_grammar_df = tabu_search_df.copy("Deep")
							optimum_grammar_mdl = best_move_mdl
							tabu_search_mdl = optimum_grammar_mdl
							tabu_list.appendleft(best_move)
							
							no_change_counter = 0
							
							break
					
						else:
							print("\t\t\tAspiration criteria not satisfied. Checking next candidate.")
							move_eval_dict.pop(best_move_index)
						
					#If best move is not tabu#
					elif tabu_flag == False:
					
						if best_move_mdl < optimum_grammar_mdl:
						
							print("\t\t\tNew Best Grammar with MDL metric of " + str(best_move_mdl))
							tabu_search_df.loc[best_move, "State"] = tabu_search_df.loc[best_move, "State"] * -1
							optimum_grammar_df = tabu_search_df.copy("Deep")
							optimum_grammar_mdl = best_move_mdl
							tabu_search_mdl = optimum_grammar_mdl
							tabu_list.appendleft(best_move)
							
							no_change_counter = 0
							
							break
							
						else:
							print("\t\t\tBest move does not form new best grammar: Current MDL is " + str(best_move_mdl) + " and Best MDL is " + str(optimum_grammar_mdl))
							
							no_change_counter += 1
							
							tabu_search_df.loc[best_move, "State"] = tabu_search_df.loc[best_move, "State"] * -1
							tabu_search_mdl = best_move_mdl		
							
							tabu_list.appendleft(best_move)
							
							break
							
				#If no current move is available:
				else:
				
					print("\t\t\tNo current best move is available. Moving to next turn without making changing state.")
					no_change_counter += 1
					break
					
			#Check stopping criteria and randomize validation for final grammar#
			if no_change_counter >= 28:
			
				print("")
				print("\t\t\tStopping criteria met. Performing random validation check.")
				move_list = move_generator_direct(tabu_search_df, Parameters.Tabu_Random_Checks, Parameters.Tabu_Direct_Move_Size)
				move_index = [x for x in range(0,Parameters.Tabu_Random_Checks-1)]
				
				pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = None)
				move_eval_list = pool_instance.map(partial(move_evaluator_direct, 
														move_list = move_list,
														full_vector_df = tabu_search_df.copy("Deep"), 
														all_indexes = all_indexes, 
														pos_unit_cost = pos_unit_cost, 
														lex_unit_cost = lex_unit_cost,
														cat_unit_cost = cat_unit_cost,
														full_cxg = full_cxg,
														current_best = best_grammar_dict["mdl_full"]
														), move_index, chunksize = 25)
				pool_instance.close()
				pool_instance.join()
					
				move_eval_dict = ct.merge(move_eval_list)
				best_move = min(move_eval_dict, key = move_eval_dict.get)
				current_best_mdl = move_eval_dict[best_move]
				best_move = move_list[best_move]
				
				if current_best_mdl < optimum_grammar_mdl:
					print("\t\tRandom state check exceeds optimum grammar. Restarting search.")
					tabu_search_df.loc[best_move, "State"] = tabu_search_df.loc[best_move, "State"] * -1
					optimum_grammar_df = tabu_search_df.copy("Deep")
					optimum_grammar_mdl = best_move_mdl
					
				else:
					print("\t\tFinal state validated. Stopping search.")
					
					break
		
		print("")
		print("\tScore before direct tabu search: " + str(starting_score_direct))
		print("\tScore after direct tabu search: " + str(optimum_grammar_mdl))
		print("\tBaseline score: " + str(baseline_mdl))
		print("")
	
		optimum_grammar_df = optimum_grammar_df[optimum_grammar_df.loc[:,"State"] == 1]
		
		return optimum_grammar_df
#-------------------------------------------------------------#