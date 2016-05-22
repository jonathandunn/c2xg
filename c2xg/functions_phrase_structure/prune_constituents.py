#---------------------------------------------------------------------------------------------#
#OUTPUT: Pruned constituents for current phrase type -----------------------------------------#
#Take possible constituents and return actual constituents for current phrase head -----------#
#---------------------------------------------------------------------------------------------#
def prune_constituents(ngram_dictionary, 
						pos_direction, 
						index_list, 
						lr_head_list, 
						rl_head_list, 
						pos_index, 
						constituent_threshold
						):

	import statistics
	
	if pos_direction == "Head-First":
		same_list = lr_head_list
		opposite_list = rl_head_list
	
	elif pos_direction == "Head-Last":
		same_list = rl_head_list
		opposite_list = lr_head_list
	
	#Remove current POS from same list#
	temp_index = same_list.index(pos_index)
	del same_list[temp_index]
	
	candidate_dictionary = {}
	
	#Hard-set frequency threshold#
	#Should be parameterized#
	#Downside to low threshold is a large number to search for later#
	mean_freq = statistics.mean(ngram_dictionary.values())
	stdev_freq = statistics.pstdev(ngram_dictionary.values())
	threshold = mean_freq + (stdev_freq * constituent_threshold)
	
	print("Threshold frequency is " + str(threshold))
		
	#Now, prune ngram sub-sequences using following criteria:#
	#First, cannot contain opposite direction head, except at edge#
	#Second, cannot contain same direction head at edge#
	#Third, can include only contiguous instances of the same pos head (e.g., NN NN is ok, but not NN IN NN)#
		
	for key in ngram_dictionary.keys():
			
		remove_flag = 0
		
		real_key = list(eval(key))
		
		if pos_direction == "Head-First":
			edge_index = len(real_key) - 1
		
		elif pos_direction == "Head-Last":
			edge_index = 0
			
		#First criteria#
		for i in range(len(real_key)):

			if real_key[i] in opposite_list and i != edge_index:
				remove_flag = 1
				
		#Second criteria#
		if real_key[edge_index] in same_list:
			remove_flag = 1
					
		#Third criteria#
		if real_key.count(pos_index) > 1:

			if pos_direction == "Head-First":
				for i in range(1, len(real_key)):
					if real_key[i] == pos_index and real_key[i - 1] != pos_index:
					
						remove_flag = 1
					
			elif pos_direction == "Head-Last":
				for i in range(len(real_key) - 1):
					if real_key[i] == pos_index and real_key[i + 1] != pos_index:

						remove_flag = 1
		
		#Fourth criteria: Frequency.#
		if ngram_dictionary[key] < threshold:
			remove_flag = 1
		
		#If not disqualified, add to pruned sequence dictionary#
		if remove_flag == 0:
			candidate_dictionary[key] = ngram_dictionary[key]
	
	return candidate_dictionary
#---------------------------------------------------------------------------------------------#