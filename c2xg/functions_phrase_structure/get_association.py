#---------------------------------------------------------------------------------------------#
#INPUT: Dictionary of pairs and their co-occurrences, unit index and frequency lists ---------#
#OUTPUT: Dataframe with co-occurrence data (a, b, c) for each pair ---------------------------#
#---------------------------------------------------------------------------------------------#
def get_association(current_pair, 
					direction, 
					base_frequency_dictionary, 
					pair_frequency_dictionary,
					total_units,
					pos_list
					):
	
	import cytoolz as ct
	from functions_phrase_structure.calculate_association import calculate_association
	
	a = ct.get(current_pair, pair_frequency_dictionary)
	b1 = (ct.get(current_pair[0], base_frequency_dictionary))
	b = b1 - a
	c1 = (ct.get(current_pair[1], base_frequency_dictionary))
	c = c1 - a
	d = total_units - a - b - c
	
	if a > b1 or a > c1:
		print("Co-occurrence error: " + str(current_pair[0]) + " : " + str(current_pair[1]))
		print("Co occurrence frequency: " + str(a))
		print("First frequency: " + str(base_frequency_dictionary[current_pair[0]]))
		print("Second frequency: " + str(base_frequency_dictionary[current_pair[1]]))
		print("")
		print("")
			
	delta_p = calculate_association(a, b, c, d, direction)
	
	return {current_pair: delta_p}
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#