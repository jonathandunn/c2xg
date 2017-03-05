#---------------------------------------------------------------------#
def save_constituents(final_grammar, Parameters, Grammar, run_parameter = 0):

	#Multi-processing safety check#
	if run_parameter == 0:
		run_parameter = 1
		#Start function proper#
		
		import random
		import codecs
		from collections import deque
		import cytoolz as ct
		import multiprocessing as mp
		from functools import partial
		
		from candidate_extraction.write_candidates import write_candidates
		from candidate_selection.move_evaluator_constituents import move_evaluator_constituents
		from candidate_selection.move_maker_constituents import move_maker_constituents
		from candidate_selection.reformat_constituents import reformat_constituents
		
		#Tabu Search for assigning sequences to phrase types#
		#--- Status {1} = ON ---#
		#--- Direction {1} = LEFT-HEAD ---#
		#--- Independence {1} = INDEPENDENT ---#
		current_head_dictionary = {}
		
		for head in range(1, len(Grammar.POS_List)):

			current_head_dictionary[head] = {}
			current_head_dictionary[head]["Status"] = -1
			current_head_dictionary[head]["Direction"] = random.randint(-1,1)
			current_head_dictionary[head]["Independence"] = random.randint(-1,1)
		
		#Because each move can change one property of a head and because each property if binary ------#
		#----- each change is evaluated for each turn and there is only one set of moves to generate---#
		move_list = []
		
		for key in current_head_dictionary:
		
			move_list.append((key, "Status"))
			move_list.append((key, "Direction"))
			move_list.append((key, "Independence"))
			
		#Make a list of tuples for each phrase structure ---------------------------------------------#
		sequence_list = []
		
		for key in final_grammar:
			
			key = eval(key)
			sequence = []
			
			for pair in key:
				sequence.append(pair[1])
			
			sequence_list.append(tuple(sequence))
	
		#--- Start main Tabu Search loop -----------------------------------------#
		#Initialize main tabu search loop#
		no_change_counter = 0
		total_loop_counter = 0
		tabu_list = deque([], maxlen = 7)
		optimum_state_score = 0.0
		best_move_score = -1
			
		while True:
		
			total_loop_counter += 1
			
			print("")
			print("\tBeginning phrase structure search loop number " + str(total_loop_counter) + " with "  + str(no_change_counter) + " loops since best state.")
			
			#Randomly choose how many changes to make in each move#
			number_list = [random.randint(1,Parameters.Tabu_Indirect_Move_Size) for x in range(1,Parameters.Tabu_Indirect_Move_Number)]
			
			pool_instance=mp.Pool(processes = Parameters.CPUs_Learning, maxtasksperchild = 1)
			move_eval_list = pool_instance.map(partial(move_evaluator_constituents, 
												head_dictionary = current_head_dictionary.copy(),
												sequence_list = sequence_list,
												current_score = best_move_score,
												move_list = move_list
												), number_list, chunksize = 1)
			pool_instance.close()
			pool_instance.join()

			move_eval_dict = ct.merge([x for x in move_eval_list])

			#----Start loop for choosing best move -------------------------------------------------------------------#
			while True:
			
				#Make sure some legal moves are left to check#
				if len(move_eval_dict) > 0:
					
					best_move = max(move_eval_dict, key = move_eval_dict.get)
					
					best_move_head_list = [x[0] for x in best_move]
					best_move_score = move_eval_dict[best_move]

					print("\t\tCurrent best move: " + str(best_move))
					
					#----------Check Tabu List--------#
					if len(tabu_list) > 0:
						flat_tabu_list = [item for sublist in tabu_list for item in sublist]
					else:
						flat_tabu_list = []
					
					tabu_flag = False
					
					for move_head in best_move_head_list:
						if move_head in flat_tabu_list:
							tabu_flag = True
							
					if tabu_flag == True:
					
						print("\t\tBest move contains head in tabu list. Checking aspiration criteria.")
						
						if best_move_score > optimum_state_score:
							
							print("\t\tBest move overcomes aspiration criteria: new best state: " + str(best_move_score))
							
							current_head_dictionary, tabu_list = move_maker_constituents(current_head_dictionary, tabu_list, best_move)

							no_change_counter = 0
							optimum_head_dictionary = current_head_dictionary.copy()
							optimum_state_score = best_move_score
							
							break
							
						#Best move is not allowed and fails aspiration criteria#
						else:
							print("\t\tBest move not allowed. Continuing search.")
							print("")
							move_eval_dict.pop(best_move)
							
					#No tabu, make the move#
					else:
						if best_move_score > optimum_state_score:
							
							print("\t\tBest move is allowed. New best score: " + str(best_move_score))
							current_head_dictionary, tabu_list = move_maker_constituents(current_head_dictionary, tabu_list, best_move)
							
							no_change_counter = 0
							optimum_head_dictionary = current_head_dictionary.copy()
							optimum_state_score = best_move_score

						else:
						
							print("\t\tBest move is allowed but doesn't reach new best state: Current = " + str(best_move_score) + " Best: " + str(optimum_state_score))
							current_head_dictionary, tabu_list = move_maker_constituents(current_head_dictionary, tabu_list, best_move)
							
							no_change_counter += 1
						
						break
							
				#Else if there are no legal moves left, do nothing#
				else:
					print("\t\tNo legal move is available. Restarting search.")
					no_change_counter += 1
					
					break
			
			#---End loop for choosing best move, returning to main tabu search loop ----------------------------------#
			
			#---Checking stopping criteria ---------------------------------------------------------------------------#
			if no_change_counter >= 14:
			
				print("")
				print("\tTabu search ending. Best score: " + str(optimum_state_score))
				
				break
				
		#--- End main Tabu Search loop -------------------------------------------#
		
		print("")
		print("")
		print("\tSearch results:")
		
		fw = codecs.open(Parameters.Debug_File + "Heads", "w", encoding = Parameters.Encoding_Type)
		
		for head in optimum_head_dictionary:
			print(Grammar.POS_List[head], end="")
			print(" , State = ", end="")
			
			fw.write(str(Grammar.POS_List[head]))
			fw.write(str(" , State = "))
			
			if optimum_head_dictionary[head]["Status"] == 1:
				print("HEAD: ", end="")
				fw.write(str("HEAD: "))
				
				if optimum_head_dictionary[head]["Direction"] == 1:
					print("Direction = Left-to-Right ", end="")
					fw.write(str("Direction = Left-to-Right "))
					
				else:
					print("Direction = Right-to-Left ", end="")
					fw.write(str("Direction = Right-to-Left "))
					
				if optimum_head_dictionary[head]["Independence"] == 1:
					print(" and Independent.")
					fw.write(str(" and Independent.\n"))
					
				else:
					print(" and Dependent.")
					fw.write(str(" and Dependent.\n"))
					
			else:
				print("NON-HEAD.")
				fw.write(str("NON-HEAD.\n"))
			
		#Now format and save constituent grammar#
		Grammar.Constituent_Dict = reformat_constituents(sequence_list, optimum_head_dictionary)
		Grammar.Type = "Constituent"
		
		write_candidates(Parameters.Data_File_Constituents, Grammar)
			
		return Grammar
#------------------------------------------------------------------------------#