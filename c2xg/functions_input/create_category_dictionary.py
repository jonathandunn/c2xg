#---------------------------------------------------------------------------------------------#
#INPUT: Filename of semantic category dictionary ---------------------------------------------#
#OUTPUT: Semantic category dictionary, from Lancaster USAS dictionary ------------------------#
#Take file and return full semantic category dictionary --------------------------------------#
#---------------------------------------------------------------------------------------------#
def create_category_dictionary(filename, encoding_type):

	semantic_category_dictionary = {}
	
	fo = open(filename, "r", encoding=encoding_type)
	
	for line in fo:
	
		line = line.replace("\n","")
		line_list = line.split(",")
		semantic_category_dictionary[line_list[0]] = line_list[1]
		
	fo.close()	
		
	return semantic_category_dictionary
#---------------------------------------------------------------------------------------------#