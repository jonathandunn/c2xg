#---------------------------------------------------------------------------------------------#
#INPUT: Semantic category dictionary ---------------------------------------------------------#
#OUTPUT: Index of semantic categories --------------------------------------------------------#
#Take file and return full semantic category dictionary --------------------------------------#
#---------------------------------------------------------------------------------------------#
def create_category_index(category_dictionary):

	category_index = []

	for word in category_dictionary.keys():
		if category_dictionary[word] not in category_index:
			category_index.append(category_dictionary[word])
			
	category_index = sorted(category_index)	

	category_index.insert(0, "n/a")
		
	return category_index
#---------------------------------------------------------------------------------------------#