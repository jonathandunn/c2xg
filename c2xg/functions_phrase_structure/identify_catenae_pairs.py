#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def identify_catenae_pairs(pair_frequency_dictionary, 
							lr_association_dictionary, 
							rl_association_dictionary,
							base_frequency_dictionary,
							pos_list
							):
	
	import time
	import cytoolz as ct
	import statistics
	from functions_phrase_structure.get_pair_status_same import get_pair_status_same
	from functions_phrase_structure.get_pair_status import get_pair_status
	from functions_phrase_structure.get_pair_head import get_pair_head
	from functions_phrase_structure.check_constraints import check_constraints
	
	time_start = time.time()
	
	#Initialize lists of pairs#
	pair_list = sorted(pair_frequency_dictionary, key = pair_frequency_dictionary.get, reverse = True)
	
	iteration_counter = 0
	end_flag = 0
	
	pair_status_dictionary = {}
	pair_head_dictionary = {}
	head_status_dictionary = {}
	same_unit_dictionary = {}
	
	#Initiate catenae-pair classifier threshold#
	catenae_threshold = max(pair_frequency_dictionary.values())
	end_threshold = min(pair_frequency_dictionary.values())
	catenae_threshold_reduction = statistics.pstdev(pair_frequency_dictionary.values()) / 10
	
	#Begin iterative learning cycle#
	while True:
	
		#Check if loop needs to continue#
		if len(pair_list) == len(list(pair_status_dictionary.keys())):
			print("All pairs accounted for: Ending Loop.")
			break
			
		else:
		
			print("Starting loop. Current threshold: " + str(catenae_threshold))
			#Start inner loop through pairs#
			for pair in pair_list:
				
				if pair[0] != 0 and pair[1] != 0:
					#For pairs where P1 = P2#
					if pair[0] == pair[1]:
					
						new_flag = 0
						pair_status = ct.get(pair[0], same_unit_dictionary, default = "None")
												
						#If necessary, check current unit P1 == P2 status and return updated dictionary#
						if pair_status == "None":

							pair_status, same_unit_dictionary = get_pair_status_same(pair,
																						pair_frequency_dictionary, 
																						lr_association_dictionary, 
																						rl_association_dictionary,
																						same_unit_dictionary
																						)

					#For all other pairs where P1 != P2#
					else:
					
						pair_status = ct.get(pair, pair_status_dictionary, default = "None")
						
						if pair_status == "None":
							new_flag = 1
							pair_status = get_pair_status(pair,
															pair_frequency_dictionary, 
															lr_association_dictionary, 
															rl_association_dictionary,
															catenae_threshold
															)
					
					#For catenae-pairs, get head pair_status and/or check consistency constraints#
					if pair_status == "Catenae" and new_flag == 1:
					
						head_status = ct.get(pair, pair_head_dictionary, default = "None")
												
						if head_status == "None":
							head_status = get_pair_head(pair,
														pair_frequency_dictionary, 
														lr_association_dictionary, 
														rl_association_dictionary
														)
						
						#Check constraints and update dictionaries#
						pair_status_dictionary, pair_head_dictionary, head_status_dictionary = check_constraints(pair, 
																													pair_status, 
																													head_status,
																													pair_status_dictionary,
																													pair_head_dictionary,
																													head_status_dictionary
																													)
						
		#print("Ending loop for threshold level " + str(catenae_threshold) + ". Currently " + str(len(list(pair_status_dictionary.keys()))) + " out of " + str(len(pair_list)) + " pairs accounted for.")
		
		if end_flag == 0:
			#Decrease catenae classifier threshold#
			catenae_threshold = catenae_threshold - catenae_threshold_reduction
			iteration_counter += 1
			#print("Loop " + str(iteration_counter) + " finished. Reducing threshold by " + str(catenae_threshold_reduction))
			print("")
		
			if catenae_threshold < 0: 
				catenae_threshold = 0
				end_flag = 1
				#print("Threshold now 0; last loop.")
		
		elif end_flag == 1:
			break
	
	#Assign remaining pairs to Non-Catenae#
	for pair in pair_frequency_dictionary.keys():
		if pair not in pair_status_dictionary:
			pair_status_dictionary[pair] = "Non-Catenae"
			
	print("Total time: " + str(time.time() - time_start))
			
	return pair_status_dictionary, pair_head_dictionary, head_status_dictionary
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#