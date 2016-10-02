#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def join_left(unit, 
				pair_status_list, 
				head_status_dictionary,
				pos_list,
				input_files,
				encoding_type,
				semantic_category_dictionary,
				word_list,
				lemma_list,
				lemma_dictionary,
				pos_dictionary,
				category_dictionary,
				delete_temp
				):
	
	import cytoolz as ct
	from collections import Counter
	
	from functions_phrase_structure.prune_unobserved import prune_unobserved
	
	direction = "L"

	active_list = [(x, y) for (x, y) in pair_status_list if x == unit]
	print(str(pos_list[unit]) + ": Initial active list contains " + str(len(active_list)) + " out of " + str(len(pair_status_list)) + " catenae pairs.")
	
	#Initiate constituent list#
	counter = 0
	constituent_list = []
	
	while True:
	
		delete_list = []
		add_list = []
		
		counter += 1
		
		#Prune unobserved sequences in current active list#
		active_list = list(set(active_list))
		
		active_list = prune_unobserved(input_files,
											active_list,
											direction,
											pos_list,
											encoding_type,
											semantic_category_dictionary,
											word_list,
											lemma_list,
											lemma_dictionary,
											pos_dictionary,
											category_dictionary,
											delete_temp
											)
											
		print("Beginning loop " + str(counter) + ": Items in active list: " + str(len(active_list)))
		
		#Loop over active list, checking each for completeness or adding candidate material#
		for i in range(len(active_list)):
		
			catenae_sequence = active_list[i]
			
			#Candidates of non-current length have reached their end#
			if len(catenae_sequence) < counter + 1:
		
				if catenae_sequence not in constituent_list:
					constituent_list.append(catenae_sequence)
					
				if i not in delete_list:
					delete_list.append(i)
			
			#Candidates which fail to project in the correct direction have reached their end#
			elif catenae_sequence[-1] in head_status_dictionary and head_status_dictionary[catenae_sequence[-1]] == "R":
				
				if catenae_sequence not in constituent_list:
					constituent_list.append(catenae_sequence)
						
				if i not in delete_list:
					delete_list.append(i)
						
			elif catenae_sequence[-1] not in head_status_dictionary:
	
				if catenae_sequence not in constituent_list:
					constituent_list.append(catenae_sequence)
						
					if i not in delete_list:
						delete_list.append(i)
			
			#Otherwise, evaluate#
			else:
				#Create list of new possible additions#

				new_unit_list = [y for (x, y) in pair_status_list if x == catenae_sequence[-1]]
															
				if len(new_unit_list) != 0:

					for new_unit in new_unit_list:
						
						temp_list = list(catenae_sequence)
						temp_list.append(new_unit)
							
						if max(Counter(temp_list).values()) == 1:
							temp_list = tuple(temp_list)
								
							if temp_list not in constituent_list and temp_list not in add_list:
								add_list.append(temp_list)
									
								if i not in delete_list:
									delete_list.append(i)									
						
		print("Adding " + str(len(add_list)) + " sequences.")
		print("Removing " +str(len(delete_list)) + " sequences.")

		#Remove completed sequences from active list#
		for i in sorted(delete_list, reverse=True):
			del active_list[i]
		#-------------------------------------------#
		
		#Now add new pairs to active list#
		active_list += add_list
		
		constituent_list = list(set(constituent_list))
		
		if len(active_list) == 0:
			print("Finished " + str(pos_list[unit]) + " from the left in " + str(counter) + " iterations. Number of constituents: " + str(len(constituent_list)))
			break
		
		try:
			print("Remaining candidates after iteration: " + str(len(active_list)))
			print("")
		
		except:
			print("Finished " + str(pos_list[unit]) + " from the left in " + str(counter) + " iterations. Number of constituents: " + str(len(constituent_list)))
			break
	
	return constituent_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#