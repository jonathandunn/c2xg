#General imports
import math
import time
import random
import os
import codecs
import platform
import statistics
import pandas as pd
import numpy as np
import cytoolz as ct
import multiprocessing as mp
from functools import partial
from collections import deque
from random import randint

#C2xG modules
from modules.process_input import create_category_dictionary
from modules.process_input import pandas_open
from modules.process_input import annotate_files
from modules.process_input import merge_conll	
from modules.process_input import merge_conll_names

from modules.candidate_extraction import write_candidates
from modules.candidate_extraction import read_candidates
from modules.candidate_extraction import create_shifted_df
from modules.candidate_extraction import get_query

from modules.constituent_reduction import expand_sentences

from modules.feature_extraction import get_query_autonomous_zero
from modules.feature_extraction import get_query_autonomous_candidate

from modules.rdrpos_tagger.Utility.Utils import readDictionary
from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger
from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import unwrap_self_RDRPOSTagger
from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import printHelp

#INPUT: Full vector DataFrame, index lists, and file name ------------------------------------#
#OUTPUT: File with readable candidate vectors ------------------------------------------------#

def write_results_pruned(full_vector_df, 
							lemma_list, 
							pos_list, 
							category_list, 
							output_file_name, 
							encoding_type
							):

	fresults = open(output_file_name, "w", encoding=encoding_type)
	fresults.write('Name,Frequency,Summed_LR,Smallest_LR,Summed_RL,Smallest_RL,Normalized_Summed_LR,Normalized_Summed_RL,Beginning_Reduced_LR,Beginning_Reduced_RL,End_Reduced_LR,End_Reduced_RL,Directional_Scalar,Directional_Categorical,Beginning_Divided_LR,Beginning_Divided_RL,End_Divided_LR,End_Divided_RL,GreatestSummed,GreatestB-Divided,GreatestE-Divided,Ranking\n')
	
	#Start loop through rows#
	for row in full_vector_df.itertuples():
		
		#First, produce readable construction representation#
		candidate_id = eval(row[1])
		candidate_str = ""
		item_counter = 0
		
		for item in candidate_id:
			item_counter += 1

			type = item[0]
			index = item[1]
			
			if type == "Lex":
				readable_item = lemma_list[index]
				readable_item = "<" + readable_item + ">"
				
			elif type == "Pos":
				readable_item = pos_list[index]
				readable_item = readable_item.upper()
				
			elif type == "Cat":
				readable_item = category_list[index]
				readable_item = readable_item.upper()
				
			if item_counter == 1:
				candidate_str += str(readable_item)
			elif item_counter > 1:
				candidate_str += " + " + str(readable_item)
				
		fresults.write('"' + candidate_str + '",')
		#Done loop to create readable construction candidate#
		
		#Second, write features values#
		fresults.write(str(row[2]) + ',')
		fresults.write(str(row[3]) + ',')
		fresults.write(str(row[4]) + ',')
		fresults.write(str(row[5]) + ',')
		fresults.write(str(row[6]) + ',')
		fresults.write(str(row[7]) + ',')
		fresults.write(str(row[8]) + ',')
		fresults.write(str(row[9]) + ',')
		fresults.write(str(row[10]) + ',')
		fresults.write(str(row[11]) + ',')
		fresults.write(str(row[12]) + ',')
		fresults.write(str(row[13]) + ',')
		fresults.write(str(row[14]) + ',')
		fresults.write(str(row[15]) + ',')
		fresults.write(str(row[16]) + ',')
		fresults.write(str(row[17]) + ',')	
		fresults.write(str(row[18]) + ',')	
		fresults.write(str(row[19]) + ',')	
		fresults.write(str(row[20]) + ',')	
		fresults.write(str(row[21]) + ',')	
		fresults.write(str(row[22]) + '\n')			

	#End loop through candidates#
	fresults.close()
	
	return
#---------------------------------------------------------------------------------------------#
#INPUT: Data required for autonomous feature extraction --------------------------------------#
#OUTPUT: Model file for feature extraction with model file and new text ----------------------#

def write_model(lemma_list, 
				pos_list, 
				word_list, 
				category_list, 
				semantic_category_dictionary, 
				sequence_list, 
				max_construction_length, 
				annotation_types, 
				candidate_list, 
				encoding_type, 
				data_file_model, 
				phrase_constituent_list, 
				lemma_dictionary, 
				pos_dictionary, 
				category_dictionary, 
				emoji_dictionary
				):

	print("Writing model file for autonomous feature extraction.")
	write_dictionary = {}
	
	write_dictionary['candidate_list'] = candidate_list
	write_dictionary['lemma_list'] = lemma_list
	write_dictionary['pos_list'] = pos_list
	write_dictionary['word_list'] = word_list
	write_dictionary['category_list'] = category_list
	write_dictionary['semantic_category_dictionary'] = semantic_category_dictionary
	write_dictionary['sequence_list'] = sequence_list
	write_dictionary['max_construction_length'] = max_construction_length
	write_dictionary['annotation_types'] = annotation_types
	write_dictionary['encoding_type'] = encoding_type
	write_dictionary['phrase_constituent_list'] = phrase_constituent_list
	write_dictionary['lemma_dictionary'] = lemma_dictionary
	write_dictionary['pos_dictionary'] = pos_dictionary
	write_dictionary['category_dictionary'] = category_dictionary
	write_dictionary['emoji_dictionary'] = emoji_dictionary
	
	write_candidates(data_file_model, write_dictionary)	
	
	return
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def write_grammar_debug(final_grammar, suffix, Grammar, Parameters):

	#Write readable grammar to file#
	debug_file = Parameters.Debug_File + suffix
	fw = codecs.open(debug_file, "w", encoding = Parameters.Encoding_Type)
	
	for construction in final_grammar:

		for unit in construction:

			type = unit[0]
			value = unit[1]
			
			if type == "Cat":
				value = "<" + str(Grammar.Category_List[value]) + ">"
			
			elif type == "Pos":
				value = str(Grammar.POS_List[value]).upper()
				
			elif type == "Lex":
				value = "'" + str(Grammar.Lemma_List[value]) + "'"
				
			fw.write(str(value) + " ")
			
		fw.write("\n")
	fw.close()
	
	return
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def tabu_search_restarts(mdl_file_list, 
							test_file_list, 
							Parameters, 
							Grammar, 
							max_candidate_length, 
							full_cxg
							):

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

	pos_unit_cost = -(math.log(2, (1/float(pos_units)))) + TOP_LEVEL_ENCODING
	lex_unit_cost = -(math.log(2, (1/float(lex_units)))) + TOP_LEVEL_ENCODING
	cat_unit_cost = -(math.log(2, (1/float(cat_units)))) + TOP_LEVEL_ENCODING
	
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
							
		#Drop frequency-weighted association measures if necessary#
		if Parameters.Use_Freq_Weighting == False:
			dropped_columns += [
								"Beginning_Divided_LR_Weighted",
								"Beginning_Divided_RL_Weighted", 
								"End_Divided_LR_Weighted", 
								"End_Divided_RL_Weighted",
								"Summed_LR_Weighted",
								"Smallest_LR_Weighted",
								"Summed_RL_Weighted", 
								"Smallest_RL_Weighted",
								"Mean_LR_Weighted", 
								"Mean_RL_Weighted", 
								"Beginning_Reduced_LR_Weighted",
								"Beginning_Reduced_RL_Weighted",
								"End_Reduced_LR_Weighted",
								"End_Reduced_RL_Weighted",
								"Directional_Scalar_Weighted",
								"Directional_Categorical_Weighted",
								"Endpoint_LR_Weighted",
								"Endpoint_RL_Weighted"
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
		
		restart_list.append(tabu_search_process(training_df, 
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
#---------------------------------------------------------------------------------------------#
#--Processing function to search through potential grammars, evaluate each, return optimum ---#

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
		
		#Initialize list of features and list of thresholds#
		feature_list = [x for x in full_vector_df.columns if x[0:5] != "Index" and x != "Candidate" and x != "Encoded" and x != "Cost" and x != "Frequency"]
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
												full_vector_df, 
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
											full_vector_df, 											 
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
			
			#print("\t\t\tMulti-processing move generation.")
			pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = 1)
			move_list = pool_instance.map(partial(move_generator, 
												grammar_dict = current_grammar_dict.copy(),
												threshold_values = threshold_values,
												checks_per_move = Parameters.Tabu_Indirect_Move_Number,
												max_move_size = Parameters.Tabu_Indirect_Move_Size
												), feature_list, chunksize = 25)
			pool_instance.close()
			pool_instance.join()
						
			move_list = [item for sublist in move_list for item in sublist]
			print("\t\t\tNumber of moves: " + str(len(move_list)))

			#----End multi-processing for move generation-------------------------------------------------------#
			
			#---------------------------------------------------------------------------------------------------#
			#Multi-process Evaluation of possible moves---------------------------------------------------------#
			print("\t\t\tMulti-processing move evaluation.")
			pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = 1)
			move_eval_list = pool_instance.map(partial(move_evaluator, 
												grammar_dict = current_grammar_dict.copy(),
												full_vector_df = full_vector_df, 
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
				#print("\t\t\tNow determining best available move.")

				#Check to ensure moves have not been exhausted#
				if len(move_eval_dict) > 0:
					
					best_move = min(move_eval_dict, key = move_eval_dict.get)
					best_move_parameters = [x[0] for x in best_move]
					
					best_move_type = best_move[0][1][1]
					best_move_mdl = move_eval_dict[best_move]

					#Check if best move is allowed by tabu list#
					#print("\t\t\t\tCandidate best move is " + str(best_move))
					
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
						#print("\t\t\t\tBest move is in tabu list. Checking aspiration criteria.")
					
						#Check if MDL score overrules tabu list#
						if best_move_mdl < best_grammar_dict["mdl_full"]:
							print("\t\t\t\tBest move is allowed: tabu overruled.")
							print("\t\t\t\tCandidate best move is " + str(best_move))
							no_best_move_flag = False
							break
							
						else:
							#print("\t\t\t\tBest move is not allowed. Moving to next best move.")
							move_eval_dict.pop(best_move)
												
					#If best move not in tabu list, stop searching#
					else:
						no_best_move_flag = False
						print("\t\t\t\tCandidate best move is " + str(best_move))
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
														full_vector_df, 													
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
			if no_change_counter >= 12:
				print("")
				print("")
				print("\t\tNo best grammar found for two cycles through tabu list. Performing validation safety check.")
				print("")
				
				#Get the score for n random states and check if any provides a better grammar#
				starting = time.time()

				best_check_grammar_dict = tabu_search_check(threshold_values, 
															Parameters.Tabu_Random_Checks, 
															full_vector_df, 
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
		best_grammar_dict, tabu_search_df = get_grammar(best_grammar_dict.copy(), 
														full_vector_df, 													
														all_indexes, 
														pos_unit_cost, 
														lex_unit_cost,
														cat_unit_cost,
														full_cxg,
														save_grammar = True
														)
		
		tabu_search_df = tabu_search_df.loc[:,["Candidate", "Encoded", "Indexes", "Cost"]]
		tabu_search_df.loc[:,"State"] = 1

		#Second, randomly initialize state by evaluating randomly generate grammars#
		print("\t\tInitializing direct tabu search.")
		
		move_list = move_generator_direct(tabu_search_df, Parameters.Tabu_Random_Checks, Parameters.Tabu_Direct_Move_Size)
		move_index = [x for x in range(0,Parameters.Tabu_Random_Checks-1)]
		
		print("\t\t\tCurrent moves: " + str(len(move_index)))
		
		pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = 1)
		move_eval_list = pool_instance.map(partial(move_evaluator_direct, 
												move_list = move_list,
												full_vector_df = tabu_search_df, 
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
		
			pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = 1)
			move_eval_list = pool_instance.map(partial(move_evaluator_direct, 
													move_list = move_list,
													full_vector_df = tabu_search_df, 
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

					#print("\t\t\tCurrent best move: " + str(best_move))
					
					tabu_flag = False
					flat_tabu_list = [item for sublist in tabu_list for item in sublist]
				
					for i in best_move:
						if i in flat_tabu_list:
							tabu_flag = True
							
					#If best move is tabu#
					if tabu_flag == True:
					
						#print("\t\t\tBest move is on tabu list. Checking aspiration criteria.")
						
						if best_move_mdl < optimum_grammar_mdl:
						
							print("\t\t\tAspiration criteria satisfied: New Best Grammar with MDL metric of " + str(best_move_mdl) + " against baseline of " + str(baseline_mdl))
							tabu_search_df.loc[best_move, "State"] = tabu_search_df.loc[best_move, "State"] * -1
							optimum_grammar_df = tabu_search_df
							optimum_grammar_mdl = best_move_mdl
							tabu_search_mdl = optimum_grammar_mdl
							tabu_list.appendleft(best_move)
							
							no_change_counter = 0
							
							break
					
						else:
							#print("\t\t\tAspiration criteria not satisfied. Checking next candidate.")
							move_eval_dict.pop(best_move_index)
						
					#If best move is not tabu#
					elif tabu_flag == False:
					
						if best_move_mdl < optimum_grammar_mdl:
						
							print("\t\t\tCurrent best move: " + str(best_move))
							print("\t\t\tNew Best Grammar with MDL metric of " + str(best_move_mdl))
							tabu_search_df.loc[best_move, "State"] = tabu_search_df.loc[best_move, "State"] * -1
							optimum_grammar_df = tabu_search_df
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
				
				pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = 1)
				move_eval_list = pool_instance.map(partial(move_evaluator_direct, 
														move_list = move_list,
														full_vector_df = tabu_search_df, 
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
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

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
	
		#First multi-process generation of n random grammars#
		print("\t\t\tGenerating random " + str(n_checks) + " grammars.")

		feature_list = [x for x in full_vector_df.columns if x[0:5] != "Index" and x != "Candidate" and x != "Encoded" and x != "Cost" and x != "Frequency"]
		ran_list = []
		for i in range(n_checks):
			ran_list.append(random.choice(feature_list))	

		pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = 1)
		grammar_list = pool_instance.map(partial(grammar_generator_random, 
													threshold_values = threshold_values,
													), ran_list, chunksize = 200)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing#
			
		#Second multi-process evaluation of grammars#
		print("\t\t\tEvaluating generated grammars.")
		pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = 1)
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
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def save_idioms(final_grammar, Parameters, Grammar):

	r = RDRPOSTagger()
			
	#Check and Change directory if necessary; only once if multi-processing#
	current_dir = os.getcwd()

	if platform.system() == "Windows":
		slash_index = current_dir.rfind("\\")
				
	else:
		slash_index = current_dir.rfind("/")
				
	current_dir = current_dir[slash_index+1:]
			
	if current_dir == "Utility":
		os.chdir(os.path.join("..", "..", "..")
	#End directory check#
			
	model_string = os.path.join("./files_data/pos_rdr/", Parameters.Language + ".RDR")
	dict_string = os.path.join("./files_data/pos_rdr/", Parameters.Language + ".DICT")
		
	r.constructSCRDRtreeFromRDRfile(model_string)
	DICT = readDictionary(dict_string)
	
	#Now tag and process each idiom#
	idiom_list = []
	sequence_list = []
	
	for sequence in final_grammar:
	
		idiom = ""
		
		for unit in sequence:
			idiom += str(Grammar.Lemma_List[unit[1]]) + " "
		
		idiom_annotated = r.tagRawSentence(DICT, idiom)
		idiom_annotated = idiom_annotated.split()

		word_list = []
		past_tag = ""
		bad_flag = 0
		
		for pair in idiom_annotated:

			pair = pair.split("/")
			word = pair[0] + " "
			tag = pair[1]
			
			word_list.append(str(word))
			
			if past_tag != "":
				if tag != past_tag:
					bad_flag = 1
					
			past_tag = tag
		
		if bad_flag == 0:

			idiom_string = "".join(word_list)
			idiom_string = idiom_string[0:len(idiom_string) -1]
			
			idiom_tagged = idiom_string.replace(" ", "_")
			
			idiom_list.append([idiom_string, idiom_tagged, tag])
			sequence_list.append(sequence)
		
	print(idiom_list)
		
	Grammar.Idiom_List = idiom_list
	
	write_grammar_debug(sequence_list, "Idioms", Grammar, Parameters)
	write_candidates(Parameters.Data_File_Idioms, Grammar)
	
	return Grammar
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def save_constructions(final_grammar, Parameters, Grammar):

	from candidate_extraction.write_candidates import write_candidates 
	
	#Now save and write grammar#
	Grammar.Construction_List = final_grammar
	write_candidates(Parameters.Data_File_Constructions, Grammar)
		
	#Write to debug file if necessary#\
	if Parameters.Debug == True:
		
		import codecs
		fw = codecs.open(Parameters.Debug_File + "Grammar." + "Constructions", "w", encoding = Parameters.Encoding_Type, errors = "replace")
			
		for construction in final_grammar:
			fw.write(str(construction) + "\n")
	
	return
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def save_constituents(final_grammar, Parameters, Grammar, run_parameter = 0):

	# NON-HEAD:
    
		# adj: adjective
		# adv: adverb	
		# aux: auxiliary verb	
		# intj: interjection
		# part: particle	
		# det: determiner
		# num: numeral
		# conj: coordinating conjunction
	
	# HEAD:
    
		# pron: pronoun
		# noun: noun
		# propn: proper noun
		# adp: adposition
		# verb: verb
		# sconj: subordinating conjunction
	
	print("Assigning heads to constituents")
	
	head_list = ["pron", "noun", "propn", "adp", "verb", "sconj"]
	non_head_list = ["adj", "adv", "aux", "intj", "part", "det", "num", "conj"]
	
	head_indexes = [Grammar.POS_List.index(x) for x in head_list if x in Grammar.POS_List]
	non_head_indexes = [Grammar.POS_List.index(x) for x in non_head_list if x in Grammar.POS_List]
	
	right_list = []
	left_list = []
	
	no_good_counter = 0
	full_counter = 0
	
	#Go through all sequences#
	for sequence in final_grammar:
	
		full_counter += 1
		
		if sequence[0][1] in non_head_indexes and sequence[-1][1] in non_head_indexes:
			#print("Not allowed: " + str(sequence))
			no_good_counter += 1
			
		elif sequence[0][1] in head_indexes and sequence[-1][1] in non_head_indexes:
			#print("Left-headed: " + str(sequence))
			left_list.append(sequence)
		
		elif sequence[0][1] in non_head_indexes and sequence[-1][1] in head_indexes:
			#print("Right-headed: " + str(sequence))
			right_list.append(sequence)
			
		elif sequence[0][1] in head_indexes and sequence[-1][1] in head_indexes:
			#print("Both end-points are heads!")
			left = head_indexes.index(sequence[0][1])
			right = head_indexes.index(sequence[-1][1])
			
			if left > right:
				#print("Left-headed: " + str(sequence))
				left_list.append(sequence)
				
			elif right > left:
				#print("Right-headed: " + str(sequence))	
				right_list.append(sequence)
				
	print("Done assigning heads: " + str(no_good_counter) + " removed out of " + str(full_counter) + " total.")
	
	write_grammar_debug(left_list, "Left-Heads." + "Constituents", Grammar, Parameters)
	write_grammar_debug(right_list, "Right-Heads." + "Constituents", Grammar, Parameters)
			
	#Now format and save constituent grammar#
	Grammar.Constituent_Dict = reformat_constituents(left_list, right_list)
	Grammar.Type = "Constituent"
	
	write_candidates(Parameters.Data_File_Constituents, Grammar)
			
	return Grammar
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def reservoir_sampling(iterator, K):
	
	result = []
	N = 0

	for item in iterator:
		N += 1
		if len( result ) < K:
			result.append( item )
		else:
			s = int(random.random() * N)
			if s < K:
				result[ s ] = item

	return result
#---------------------------------------------------------------------------------------------#
#INPUT: Phrase constituent list, lists of heads in both directions ---------------------------#
#OUTPUT: Updated direction-specific constituents organized by ngram length--------------------#

def reformat_constituents(left_list, right_list):

	lr_constituent_dictionary = {}
	rl_constituent_dictionary = {}
	
	#Simplify from (Pos, Index) tuples to lists of indexes#
	left_list = [[x[1] for x in y] for y in left_list]
	right_list = [[x[1] for x in y] for y in right_list]
	
	#Begin loop through identified constituents#
	for rule in left_list:
	
		current_head = rule[0]
			
		try:
			lr_constituent_dictionary[current_head].append(rule)
							
		except:
			lr_constituent_dictionary[current_head] = []
			lr_constituent_dictionary[current_head].append(rule)
			
		
	for rule in right_list:
			
		current_head = rule[-1]
			
		try:
			rl_constituent_dictionary[current_head].append(rule)
								
		except:
			rl_constituent_dictionary[current_head] = []
			rl_constituent_dictionary[current_head].append(rule)
						
	return [lr_constituent_dictionary, rl_constituent_dictionary]
#---------------------------------------------------------------------------------------------#
#INPUT: Candidate vector dataframe pruned by association strength ----------------------------#
#OUTPUT: Candidate vector dataframe pruned horizontally --------------------------------------#

def prune_horizontal(full_vector_df):
    
	delete_list = []
	
	sort_df = full_vector_df.loc[:, ["Candidate", "End_Divded_RL"]]
	sort_df = sort_df.sort_values(by = "Candidate", ascending = False, inplace = False)
		
	last = sort_df.iloc[0]
	
	for i in range(1, sort_df.shape[0]):
		
		current = sort_df.iloc[i]
		
		current_candidate = str(current[0])
		current_candidate = current_candidate[1:len(current_candidate)-1]
		
		last_candidate = str(last[0])
		last_candidate = last_candidate[1:len(last_candidate)-1]
		
		
		if current_candidate in last_candidate:
			delete_list.append(current.name)
		
		last = sort_df.iloc[i]
			
	if len(delete_list) > 0:	
		pruned_vector_df = full_vector_df[~full_vector_df.index.isin(delete_list)]
	
	else:
		pruned_vector_df = full_vector_df

	return pruned_vector_df
#---------------------------------------------------------------------------------------------#
#Get indexes covered by each construction ----------------------------------------------------#

def process_coverage(Parameters, 
						Grammar, 
						training_testing_files,
						testing_files,
						max_candidate_length, 
						candidate_list_formatted, 
						association_df, 
						expand_flag, 
						run_parameter = 0
						):
	
	#Prevent pool workers from starting here#
	if run_parameter == 0:
	#---------------------------------------#
		run_parameter = 1
		
		#-----Combine training_testing_files into one file for each restart and combine and testing_files into one file --#
		#-------This makes testing and restarts easier to handle ---------------------------------------------------------#
		
		print("\t\tJoining training-testing and testing files into single file for each restart with a single file for testing.")
		tuple_list, training_list, testing_list = merge_conll_names(training_testing_files, testing_files, Parameters)	

		#Start multi-processing#
		pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = 1)
		pool_instance.map(partial(merge_conll, 
									encoding_type = Parameters.Encoding_Type
									), tuple_list, chunksize = 1)
		pool_instance.close()
		pool_instance.join()
		#End multi-processing#

		#----FINISHED CONSOLIDATING TESTING SETS -----------------------------------------------------------------#
		
		#First, evaluate string candidates to lists and sort by length#
		eval_list = ct.groupby(len, candidate_list_formatted)
		for_list = training_list + testing_list
		
		for input_file in for_list:
		
			print("")
			print("\t\tBeginning MDL-prep for " + str(input_file))
			print("")
			
			input_file_original = input_file
			
			#First, load and expand input file#
			input_df = pandas_open(input_file, 
									Parameters, 
									Grammar,
									write_output = False,
									delete_temp = False
									)
												
			total_words = len(input_df)
												
			if expand_flag == True:
				input_df = expand_sentences(input_df, Grammar, write_output = False)
					
			else:
				input_df.loc[:,"Alt"] = 0
				input_df = input_df.loc[:,['Sent', 'Alt', 'Mas', "Lex", 'Pos', 'Cat']]

			#Save DF for later calculations#
			write_candidates(input_file + ".Training", input_df)
			
			#Second, call search function by length#
			result_list = []

			if input_df.empty:
				print("ERROR: Training DataFrame is empty.")
				sys.kill()
	
			for i in eval_list.keys():
					
				current_length = i
				current_list = eval_list[i]
	
				if current_list:
								
					current_df = input_df
								
					print("")
					print("\t\t\tStarting constructions of length " + str(i) + ": " + str(len(current_list)))
								
					if current_length > 1:
							
						#Create shifted alt-only dataframe for length of template#
						alt_columns = []
						alt_columns_names = []
						for i in range(current_length):
							alt_columns.append(1)
							alt_columns_names.append("c" + str(i))
							
						alt_dataframe = create_shifted_df(current_df, 1, alt_columns)
						alt_dataframe.columns = alt_columns_names
								
						query_string = get_query(alt_columns_names)
						row_mask_alt = alt_dataframe.eval(query_string)
						del alt_dataframe
						
						#Create shifted sent-only dataframe for length of template#
						sent_columns = []
						sent_columns_names = []
						for i in range(current_length):
							sent_columns.append(0)
							sent_columns_names.append("c" + str(i))
							
						sent_dataframe = create_shifted_df(current_df, 0, sent_columns)
						sent_dataframe.columns = sent_columns_names
						query_string = get_query(sent_columns_names)
						row_mask_sent = sent_dataframe.eval(query_string)
						del sent_dataframe
								
						#Create and shift template-specific dataframe#
						current_df = create_shifted_length_df(current_df, current_length)

						current_df = current_df.loc[row_mask_sent & row_mask_alt,]
						del row_mask_sent
						del row_mask_alt
							
						#Remove NaNS and change dtypes#
						current_df.fillna(value=0, inplace=True)
						column_list = current_df.columns.values.tolist()
						current_df = current_df[column_list].astype(int)
							
					elif current_length == 1:
							
						query_string = "(Alt == 0)"
						current_df = current_df.query(query_string, parser='pandas', engine='numexpr')
						current_df = current_df.loc[:,['Sent', "Lex", 'Pos', 'Cat']]
						current_df.columns = ['Sent', 'Lem0', 'Pos0', 'Cat0']
						
					#Remove zero valued indexes#
					column_list = current_df.columns.values.tolist()
					query_string = get_query_autonomous_zero(column_list)
					current_df = current_df.query(query_string, parser='pandas', engine='numexpr')

					#Now, search for individual sequences within prepared DataFrame#
					#Start multi-processing#
					pool_instance=mp.Pool(processes = Parameters.CPUs_General, maxtasksperchild = 1)
					coverage_list = pool_instance.map(partial(get_coverage, 
																current_df = current_df, 
																lemma_list = Grammar.Lemma_List, 
																pos_list = Grammar.POS_List, 
																category_list = Grammar.Category_List,
																total_words = total_words
																), [x for x in current_list], chunksize = 500)
					pool_instance.close()
					pool_instance.join()
					#End multi-processing#

					coverage_list = ct.merge([x for x in coverage_list])
					result_list.append(coverage_list)
	
					del current_df
				
			#Merge and save coverage dictionaries for each training set#
			result_dict = ct.merge([x for x in result_list])

			del result_list

			result_dict_encoded = {}
			result_dict_indexes = {}

			if len(result_dict.keys()) == 0:
			
				print("ERROR: No candidate coverage results for this datset.")
			
			else:
				for key in result_dict:

					result_dict_encoded[key] = result_dict[key]["Encoded"]
					result_dict_indexes[key] = result_dict[key]["Indexes"]
				
				del result_dict

				result_df_encoded = pd.DataFrame.from_dict(result_dict_encoded, orient = "index")
				result_df_encoded.columns = ["Encoded"]
				
				result_df_indexes = pd.DataFrame.from_dict(result_dict_indexes, orient = "index", dtype = "object")
				result_df_indexes.columns = ["Indexes"]
				
				result_df = pd.merge(result_df_encoded, result_df_indexes, left_index = True, right_index = True)
				index_list = [str(list(x)) for x in result_df.index.tolist()]
				result_df.loc[:,"Candidate"] = index_list

				del result_df_encoded
				del result_df_indexes
				
				result_df = pd.merge(result_df, association_df, on = "Candidate")

				write_candidates(input_file + ".MDL", result_df)
				
		training_list = [x for x in training_list]
		testing_list = [x for x in testing_list]
		
		return training_list, testing_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def move_maker_constituents(current_head_dictionary, tabu_list, best_move):

	head_list = []
	for move in best_move:
	
		move_head = move[0]
		move_type = move[1]
		head_list.append(move_head)
		
		current_head_dictionary[move_head][move_type] = current_head_dictionary[move_head][move_type] * -1
	
	tabu_list.appendleft(head_list)

	return current_head_dictionary, tabu_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def move_generator_direct(tabu_search_df, checks_per_move, max_move_size):

	move_list = []
	index_list = list(tabu_search_df.index.values)
	
	for i in range(checks_per_move):
	
		move_size = random.randint(1,max_move_size)
		
		if move_size < len(index_list):
			move_list.append(random.sample(index_list, move_size))
			
		else:
			move_list.append(index_list)

	return move_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def move_generator(current_feature, 
					grammar_dict, 
					threshold_values, 
					checks_per_move, 
					max_move_size
					):

	return_list = []
	move_list = []
	
	#FIRST, GET OR CANDIDATES, HALF UP FROM CURRENT AND HALF DOWN#
	checks_per_move_or = 2
	checks_per_move = checks_per_move - (checks_per_move_or * 2)
	
	#If feature is currently on, randomly find thresholds above and below current#
	if current_feature in grammar_dict:
		
		current_threshold = grammar_dict[current_feature][0]
		current_threshold_values = threshold_values[current_feature]
			
	#IF a feature has no current threshold, find the median value#
	else:
		current_threshold_values = threshold_values[current_feature]
		current_threshold = statistics.median(current_threshold_values)
		
	#Sometimes a parameter has fewer settings than available moves#
	if checks_per_move_or > len(current_threshold_values):
		move_list = current_threshold_values
		
	#Usually not#
	else:
		
		#Get as many values above / below threshold as possible#
		above_values = [x for x in current_threshold_values if x > current_threshold]
		below_values = [x for x in current_threshold_values if x < current_threshold]
			
		if len(above_values) < checks_per_move_or:
			move_list += above_values
		
		else:
			move_list += random.sample(above_values, checks_per_move_or)
					
		if len(below_values) < checks_per_move_or:
			move_list += below_values
					
		else:
			move_list += random.sample(below_values, checks_per_move_or)
			
	#Add OR moves to list#			
	for move in move_list:
		return_list.append({current_feature: (move, "OR")})
		
	#Add an "OFF" move for current feature#
	return_list.append({current_feature: (0, "OFF")})
	
	#SECOND, GET RANDOM MOVES CONTAINING THE CURRENT FEATURE AMONG OTHERS#
	for i in range(checks_per_move):
		
		fixed_size = type = random.randint(2,max_move_size)
		move_dict = grammar_generator_random(current_feature, threshold_values, random_state = False, fixed_size = fixed_size)
		return_list.append(move_dict)

	return return_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

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

	#Get current reference move#
	current_move = move_list[current_index]
	
	#Reverse indexes in current move by multiplying by -1 #
	full_vector_df.loc[current_move, "State"] = full_vector_df.loc[current_move, "State"] * -1
	full_vector_df = full_vector_df[full_vector_df.State != -1]

	#Evaluate for MDL metric#
	mdl_l1, mdl_l2, mdl_full = grammar_evaluator(full_vector_df.loc[:,["Candidate", "Encoded", "Indexes", "Cost"]], all_indexes)
								
	if mdl_full != 1000000000000000000000 and mdl_full != current_best and mdl_full != "nan":	
								
		return {current_index: mdl_full}
		
	else:
		return {}
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def move_evaluator_constituents(move_size, 
									move_list, 
									sequence_list, 
									head_dictionary, 
									current_score
									):

	#---Randomly change N items from move list------------------------------------------------#
	try:
		current_move_list = random.sample(move_list, move_size)
	
	except:
		current_move_list = move_list

	for move in current_move_list:
	
		head = move[0]
		parameter = move[1]
	
		#Reverse the parameter for the head specified in the current move#
		head_dictionary[head][parameter] = head_dictionary[head][parameter] * -1
	#---Done randomly selecting moves----------------------------------------------------------#
	
		allowed_sequences = check_constituent_constraints(sequence_list, head_dictionary)

		if len(allowed_sequences) > 0:

			score = float(len(allowed_sequences)) / float(len(sequence_list))
			
			if score != current_score:
				return {tuple(current_move_list): score}
					
		return {tuple(current_move_list): 0.0}
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

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
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def merge_and_save(grammar_type, 
					fold_results, 
					Parameters, 
					Grammar, 
					input_files = None
					):
	
	#First, Average MDL scores and return merged grammar#
	mdl_dict = {}
	grammar_dict = {}

	counter = 0
		
	for fold_file in fold_results:
		
		counter += 1
		current_grammar, current_mdl = read_candidates(fold_file)
			
		mdl_dict[counter] = current_mdl
		grammar_dict[counter] = current_grammar
			
	mdl_list = list(mdl_dict.values())
	average_mdl = sum(mdl_list) / len(mdl_list)
		
	final_grammar = set(grammar_dict[1].tolist())
		
	for key in grammar_dict:
		
		final_grammar = final_grammar.union(set(grammar_dict[key].tolist()))
		
	final_grammar = [eval(x) for x in final_grammar]
	final_grammar = horizontal_pruning(final_grammar)
			
	#Print final merged results#	
	print("\tCross-fold MDL for " + str(grammar_type) + ": " + str(average_mdl))
	print("\tLength of merged grammar: " + str(len(final_grammar)))
	
	write_grammar_debug(final_grammar, "Full." + str(grammar_type), Grammar, Parameters)
	
	#Now save the final grammar#
	print("")
	print("Saving final grammar: " + str(grammar_type))
	
	if grammar_type == "Idiom":
		Grammar = save_idioms(final_grammar, Parameters, Grammar)
		Grammar.Type = "Idiom"
		
	elif grammar_type == "Constituent":
		Grammar = save_constituents(final_grammar, Parameters, Grammar)
		Grammar.Type = "Constituent"
		
	elif grammar_type == "Construction":
		Grammar = save_constructions(final_grammar, Parameters, Grammar)
		Grammar.Type = "Construction"
		
	#Delete fold result files if necessary#
	if Parameters.Delete_Temp == True:
				
		print("\tDeleting temp files.")
		from process_input.check_data_files import check_data_files
			
		if input_files != None:
			fold_results += [x.replace("Temp/","Temp/Candidates/") + ".Candidates." + type + "s" for x in input_files]
			
		if Parameters.Run_Tagger == True:
			fold_results += input_files
			
		for file in fold_results:
			check_data_files(file)
				
	return Grammar
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def horizontal_pruning(final_grammar):

	pruned_grammar = []

	for construction1 in final_grammar:
	
		remove_flag = 0

		for construction2 in final_grammar:
			
			if construction2 != construction1 and len(construction2) > len(construction1):
				
				#Left check#
				if construction1 == construction2[0:len(construction1)]:
					remove_flag = 1
					
				#Right check#
				elif construction1 == construction2[-(len(construction1)):]:
					remove_flag = 1
					
		if remove_flag == 0:
			pruned_grammar.append(construction1)
			
	print("Length of unpruned grammar: " + str(len(final_grammar)))
	print("Length of pruned grammar: " + str(len(pruned_grammar)))		

	return pruned_grammar
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def grammar_query(grammar_dict):
	
	and_counter = 0
	or_counter = 0
	
	and_query = "("
	or_query = "("
	
	for feature in grammar_dict.keys():
	
		if feature not in ["Candidate", "Frequency", "Encoded", "Indexes", "mdl_l1", "mdl_l2", "mdl_l3", "mdl_full", "grammar_size"]:
			
			current_threshold = str(grammar_dict[feature][0])
			current_type = str(grammar_dict[feature][1])

			if current_type == "AND":
					
					and_counter += 1
					
					if and_counter > 1:
						and_query += " & "
						
					and_query += str(feature) + " > " + str(current_threshold)
									
			elif current_type == "OR":
					
					or_counter += 1
					
					if or_counter > 1:
						or_query += " | "
						
					or_query += "(" + str(feature) + " > " + str(current_threshold) + ")"
					
	query_string = ""
	
	if and_counter >= 1:
		and_query += ")"
		query_string += and_query
		
	if or_counter >= 1:
		or_query += ")"
		
		if and_counter >= 1:
			query_string += " | " + or_query
		
		else:
			query_string += or_query
	
	return query_string
#---------------------------------------------------------------------------------------------#
#--Take dictionary of possible feature weights -----------------------------------------------#
#--Return random grammar ---------------------------------------------------------------------#

def grammar_generator_random(current_feature, 
								threshold_values, 
								random_state = True, 
								fixed_size = 0
								):
	
	feature_list = list(threshold_values.keys())
	
	if fixed_size == 0:
		grammar_size = random.randint(1,len(feature_list))
	else:
		grammar_size = fixed_size
			
	grammar_list = reservoir_sampling(feature_list, grammar_size)
		
	if current_feature not in grammar_list:
		grammar_list.append(current_feature)
		
	grammar_dict = {}
		
	for feature in grammar_list:

		if random_state == True:
			type = random.randint(0,1)
			if type == 0:
				type = "AND"
			elif type == 1:
				type = "OR"
			
		else:
			type = "AND"
			
		temp_parameter_list = list(threshold_values[feature])
		temp_parameter_value = random.choice(temp_parameter_list)
		grammar_dict[feature] = (temp_parameter_value, type)
			
	return grammar_dict
#---------------------------------------------------------------------------------------------#
#--Take dictionary of initial feature weights, Generate random combination of features -------#
#--Return string for pandas query ------------------------------------------------------------#

def grammar_generator(threshold_dict):
		
	feature_list = list(threshold_dict.keys())
	grammar_size = randint(1,len(feature_list))
	grammar_list = reservoir_sampling(feature_list, grammar_size)
	
	grammar_dict = {}
	
	for feature in grammar_list:
		
		type = randint(0,1)
		if type == 0:
			type = "AND"
		elif type == 1:
			type = "OR"
		
		grammar_dict[feature] = (threshold_dict[feature], type)

	return grammar_dict
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def grammar_evaluator_baseline(test_df):
	
	TOP_LEVEL_ENCODING = 0.301
	
	#Limit test file to only original units (no complex constituents)#
	test_df = test_df.loc[test_df.loc[:,"Alt"] == 0]
	
	#Calculate encoding size#
	number_of_units = len(test_df)
	unit_cost = -(math.log(2, (1/number_of_units))) + TOP_LEVEL_ENCODING
	total_unencoded_size = unit_cost * number_of_units
	
	return total_unencoded_size
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def grammar_evaluator(grammar_df, all_indexes):

	if len(grammar_df) > 2:
	
		TOP_LEVEL_ENCODING = 0.301

		#L1 is the encoding size of the grammar: only the cost of encoding the current construction grammar-------#
		#Regret (residual unencoded units) is included in L3 -----------------------------------------------------#
		
		#Sum cost of constructions in current grammar#
		mdl_l1 = grammar_df.loc[:,"Cost"].sum()
		
		#L2 contains, first, the cost of encoding all constructions#
		num_constructions_encoded = grammar_df.loc[:,"Encoded"].sum()
		num_constructions = len(grammar_df)
		
		cost_per_construction = -(math.log(2, (1/float(num_constructions)))) + TOP_LEVEL_ENCODING
		
		mdl_l2_constructions = cost_per_construction * num_constructions_encoded
		
		#Find indexes not encoded by a construction, from grammar_df#
		#Combine and find set of all tuples in "Indexes" #
		encoded_indexes = set([item for sublist in grammar_df.loc[:,"Indexes"].tolist() for item in sublist])
		unencoded_indexes = set(all_indexes) - set(encoded_indexes)
		
		#L2 contains, second, the regret of all unencoded indexes#
		number_unencoded = len(list(unencoded_indexes))
		unencoded_cost = -(math.log(2, (1/float(number_unencoded)))) + TOP_LEVEL_ENCODING
		
		mdl_l2_unencoded = number_unencoded * unencoded_cost
		
		#Sum final MDL metric#
		mdl_l2 = mdl_l2_constructions + mdl_l2_unencoded
		mdl_full = mdl_l1 + mdl_l2
		
	else:
	
		mdl_l1 = 100000000000000000
		mdl_l2 = 100000000000000000
		mdl_full = 100000000000000000000000
	
	return mdl_l1, mdl_l2, mdl_full
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def get_threshold_ranges(feature_list, full_vector_df, number_thresholds, max_candidate_length):

	threshold_values = {}
	
	for feature in feature_list:
		
		if feature == "Directional_Categorical_Unweighted":
		
			threshold_list = [x for x in range(1, max_candidate_length)]
			threshold_values[feature] = threshold_list
		
		else:
		
			#Set upper and lower bounds on thresholds, and find increment#
			min_value = full_vector_df.loc[:,feature].min(skipna = True)
			max_value = full_vector_df.loc[:,feature].max(skipna = True)
			increment = (max_value - min_value) / float(number_thresholds)
			
			#Initialize for search#
			current_threshold = max_value - 0.0001
			
			threshold_list = [current_threshold]
			
			for i in range(number_thresholds):
				current_threshold = current_threshold - increment
				threshold_list.append(current_threshold)
				
			threshold_values[feature] = threshold_list
			
	return	threshold_values
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def get_grammar(grammar_dict,
					full_vector_df, 					 
					all_indexes, 
					pos_unit_cost, 
					lex_unit_cost,
					cat_unit_cost,
					full_cxg, 
					save_grammar = False
					):

	query_string = grammar_query(grammar_dict)
	
	if len(full_vector_df) > 2:
	
		try:
			full_vector_df = full_vector_df.query(query_string, parser = "pandas", engine = "numexpr")
			
		except:
			print("get_grammar.py line 25")
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
#---------------------------------------------------------------------------------------------#
#INPUT: Current template and DataFrame -------------------------------------------------------#
#OUTPUT: DataFrame with matches to template --------------------------------------------------#

def get_coverage(candidate, 
				current_df, 
				lemma_list, 
				pos_list, 
				category_list,
				total_words
				):

	current_length = len(candidate)
	coverage_dictionary = {}
	coverage_dictionary[tuple(candidate)] = {}
	
	candidate_query = get_query_autonomous_candidate(candidate)
	search_df = current_df.query(candidate_query, parser='pandas', engine='numexpr')

	#Find duplicated rows within same sentence and remove those which are duplicated#
	column_list = search_df.columns.values.tolist()
	row_mask = search_df.duplicated(subset=column_list, keep="first")
	search_df = search_df.loc[~row_mask,]
	del row_mask
	
	list_of_indexes = []
	times_encoded = 0
	
	#If no matches, just use empty series#
	if len(search_df) > 0:
			
		search_df = search_df.drop_duplicates(subset = 'Mas', keep = "first")
		search_df = search_df.loc[:,["Mas", "EndMas"]]
		
		for row in search_df.itertuples(index = False, name = None):

			times_encoded += 1
			list_of_indexes += [x for x in range(row[0], row[1] + 1)]
			
	else:
	
		times_encoded = 0
		list_of_indexes = ()
		
	candidate_key = tuple(candidate)
	list_of_indexes = [tuple(list_of_indexes)]

	coverage_dictionary[candidate_key]["Encoded"] = times_encoded
	coverage_dictionary[candidate_key]["Indexes"] = list_of_indexes
	
	return coverage_dictionary
#---------------------------------------------------------------------------------------------#
#INPUT: template -----------------------------------------------------------------------------#
#OUTPUT: Take a dataframe, the column to repeat, and a listof times to repeat ----------------#
#Specific to creating alt / sent dataframes b/c more efficient than a generalized version ----#

def create_shifted_length_df(original_df, current_length):
	
	column_list = []
	
	ordered_columns = []
	named_columns = []
	
	ordered_columns.append(['Sent', 'Mas'])
	named_columns.append('Sent')
	named_columns.append('Mas')
	
	for i in range(current_length):
		ordered_columns.append(["Lex", 'Pos', 'Cat'])
		named_columns.append("Lex" + str(i))
		named_columns.append('Pos' + str(i))
		named_columns.append('Cat' + str(i))
		
	for i in range(len(ordered_columns)):
		holder_df = original_df.loc[:,ordered_columns[i]]
		column_list.append(holder_df.shift(-i))
		del holder_df

		if i == len(ordered_columns)-1:
			holder_df = original_df.loc[:,"Mas"]
			column_list.append(holder_df.shift(-i))
			named_columns.append('EndMas')
			
	
	original_df = pd.concat(column_list, axis=1)
	del column_list
	
	original_df.columns = named_columns
	
	return original_df
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def construction_cost(training_df, slot_r_cost, pos_unit_cost, lex_unit_cost, cat_unit_cost):

	cost_list = []
	
	for current_row in training_df.loc[:,"Candidate"].iteritems():
		
		construction = eval(current_row[1])
		cost = 0.0
		
		for slot in construction:
			
			slot_type = slot[0]
			cost += slot_r_cost
			
			if slot_type == "Pos":
				cost += pos_unit_cost
				
			elif slot_type == "Lex":
				cost += lex_unit_cost
				
			elif slot_type == "Cat":
				cost += cat_unit_cost
				
		cost_list.append(cost)
		
	training_df.loc[:,"Cost"] = cost_list
	
	return training_df
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

def check_constituent_constraints(sequence_list, head_dictionary):

	total_sequences =  len(sequence_list)
	allowed_sequences = []
	
	#Now determine which sequences are well-formed#
	for sequence in sequence_list:
	
		left_head = sequence[0]
		right_head = sequence[-1]
		
		#Check if the left end-point is filled by a left head#
		if head_dictionary[left_head]["Status"] == 1:
			if head_dictionary[left_head]["Direction"] == 1:
			
				#Possible left-headed phrase: Check constraints#
				
				#-----Right endpoint not a head -----------------------or not a left head--------------------------or it is independent ------------------------#
				if head_dictionary[right_head]["Status"] != 1 or head_dictionary[right_head]["Direction"] != 1 or head_dictionary[right_head]["Independence"] == 1:
					allowed_sequences.append(sequence)
					#print("Left-headed: " + str(sequence))
			
		#Otherwise check if the right end-point is filled by a right head#
		elif head_dictionary[right_head]["Status"] == 1:
			if head_dictionary[right_head]["Direction"] == -1:
			
				#Possible right-headed phrase: Check constraints#
				
				#-----Left endpoint not a head -----------------------or not a right head--------------------------or it is independent ------------------------#
				if head_dictionary[left_head]["Status"] != 1 or head_dictionary[left_head]["Direction"] != -1 or head_dictionary[left_head]["Independence"] == 1:
					allowed_sequences.append(sequence)
					#print("Right-headed: " + str(sequence))
					
	return allowed_sequences
#---------------------------------------------------------------------------------------------#
#INPUT: Current candidates, phrase constituents, and expanded DataFrame filenames ------------#
#OUTPUT: Candidates with phrase constituents and their frequency added -----------------------#

def add_constituent_candidates(full_candidate_list, phrase_constituent_list):
	
	for i in range(len(phrase_constituent_list)):
		current_dictionary = phrase_constituent_list[i]
		
		for key in current_dictionary.keys():
			current_list = current_dictionary[key]

			
			for sequence in current_list:
				current_construction = []
				
				for unit in sequence:
					current_construction.append(('Pos', unit))
					
				current_label = str(current_construction)
				
				if current_label not in full_candidate_list:
					full_candidate_list.append(current_label)
					
	return full_candidate_list
#---------------------------------------------------------------------------------------------#