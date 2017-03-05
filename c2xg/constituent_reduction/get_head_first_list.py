#---------------------------------------------------------------------------------------------#
#INPUT: Dictionary with keys head index and values length of constituent----------------------#
#OUTPUT: List for removal --------------------------------------------------------------------#
#Produce reduction info for head-first phrases -----------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_head_first_list(remove_dictionary):

	#Create list of indexes to be removed#
	remove_list = []
	head_list = list(remove_dictionary.keys())
			
	for i in range(len(head_list)):
	
		current_length = remove_dictionary[head_list[i]]
		
		for j in range(1,current_length):
			remove_list.append(head_list[i] + j)
				
	return remove_list, head_list
#---------------------------------------------------------------------------------------------#