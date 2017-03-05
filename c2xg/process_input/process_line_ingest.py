#---------------------------------------------------------------------------------------------#
#INPUT: Line of text from Malt-Parser formatted corpus and semantic category dictionary ------#
#OUTPUT: Dictionary of representations for current word --------------------------------------#
#Take line, containing multiple representations of a single word, return dictionary ----------#
#---------------------------------------------------------------------------------------------#
def process_line_ingest(line, semantic_category_dictionary):

	import cytoolz as ct
	
	semantic_category = ct.get(line[0].lower(), semantic_category_dictionary, default="n/a")
	line.insert(4, semantic_category.lower())
		
	return line
#---------------------------------------------------------------------------------------------#