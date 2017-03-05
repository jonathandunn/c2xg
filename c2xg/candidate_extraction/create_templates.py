#---------------------------------------------------------------------------------------------#
#INPUT: Max construction length, annotation types, and lists of allowable elements for each --#
#OUTPUT: List of templates for possible constructions ----------------------------------------#
#---------------------------------------------------------------------------------------------#
def create_templates(annotation_types, max_construction_length):

	import itertools
	
	sentence_list_candidates = {}
	current_candidates = {}
	sequence_list = []
	counter = 0
	progress_counter = 0
	
	#First, generate all possible templates (e.g., Word-Form, POS, Role, Word-Form)#
	for ngram in range(2,max_construction_length + 1):
		for p in itertools.product(annotation_types, repeat=ngram):
			sequence_list.append(p)
			
	print("")
	print("Number of templates: " + str(len(sequence_list)))
	print("")
			
	return sequence_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#