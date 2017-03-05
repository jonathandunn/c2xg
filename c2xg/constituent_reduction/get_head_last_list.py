#---------------------------------------------------------------------------------------------#
#INPUT: Dictionary with keys head index and values length of constituent----------------------#
#OUTPUT: List for removal --------------------------------------------------------------------#
#Produce reduction info for head-last phrases ------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_head_last_list(remove_dictionary):

	#Create list of indexes to be removed#
	remove_list = []
	head_list = []
	
	start_list = list(remove_dictionary.keys())
	
	for i in range(len(start_list)):
	
		current_length = remove_dictionary[start_list[i]]

		for j in range(0, current_length-1):
			remove_list.append(start_list[i] + j)

		head_list.append(start_list[i] + current_length-1)
				
	return remove_list, head_list
#---------------------------------------------------------------------------------------------#