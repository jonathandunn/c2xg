#---------------------------------------------------------------------------------------------#
#INPUT: List of units to find, index to find them in -----------------------------------------#
#OUTPUT: List unit indexes -------------------------------------------------------------------#
#Take list of units and list of indexes and return list of indexes of requested units --------#
#---------------------------------------------------------------------------------------------#
def find_unit_index(list_of_units, index_list):

	list_of_indexes = list_of_units
	
	for i in range(len(index_list)):
		if index_list[i] in list_of_units:
			location = list_of_units.index(index_list[i])
			list_of_indexes[location] = i
			
	return list_of_indexes
#---------------------------------------------------------------------------------------------#