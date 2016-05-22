#---------------------------------------------------------------------------------------------#
#INPUT: List of dictionaries with unit frequencies--------------------------------------------#
#OUTPUT: List of lists of allowed units-------------------------------------------------------#
#Take dictionary of elements and return index lists ------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_index_lists(full_dictionary):

	new_dictionary = {}
	
	#First, separate label dictionaries with frequency info#
	lemma_dictionary = full_dictionary['lemma']
	pos_dictionary = full_dictionary['pos']
	word_dictionary = full_dictionary['word']
	category_dictionary = full_dictionary['category']	
		
	#Second, create lists of label indexes#
	word_list = sorted(word_dictionary.keys())
	lemma_list = sorted(lemma_dictionary.keys())
	pos_list = sorted(pos_dictionary.keys())
	category_list = sorted(category_dictionary.keys())	
	
	word_list.insert(0, "n/a")	
	lemma_list.insert(0, "n/a")
	pos_list.insert(0, "n/a")
	
	temp_index = category_list.index("n/a")
	del category_list[temp_index]
	category_list.insert(0, "n/a")
	
	word_frequency = word_dictionary
	lemma_frequency = lemma_dictionary
	pos_frequency = pos_dictionary
	category_frequency = category_dictionary
		
	#Fifth, append items onto a single list for returning and writing#	
	new_dictionary['lemma_list'] = lemma_list
	new_dictionary['pos_list'] = pos_list
	new_dictionary['word_list'] = word_list
	new_dictionary['category_list'] = category_list
	
	new_dictionary['lemma_frequency'] = lemma_frequency
	new_dictionary['pos_frequency'] = pos_frequency
	new_dictionary['word_frequency'] = word_frequency
	new_dictionary['category_frequency'] = category_frequency
	
	lemma_dictionary = {}
	pos_dictionary = {}
	category_dictionary = {}
	
	for i in range(len(lemma_list)):
		lemma_dictionary[lemma_list[i]] = i
		
	for i in range(len(pos_list)):
		pos_dictionary[pos_list[i]] = i
		
	for i in range(len(category_list)):
		category_dictionary[category_list[i]] = i
		
	new_dictionary['lemma_dictionary'] = lemma_dictionary
	new_dictionary['pos_dictionary'] = pos_dictionary
	new_dictionary['category_dictionary'] = category_dictionary
			
	return new_dictionary
#---------------------------------------------------------------------------------------------#