#---------------------------------------------------------------------------------------------#
#INPUT: Filename storing Dataframe with all elements and a single template -------------------#
#OUTPUT: Candidates from all files for current template that are above frequency threshold ---#
#Process single template and return first-cut candidates--------------------------------------#
#---------------------------------------------------------------------------------------------#
def templates_to_candidates(current_df, 
								filename, 
								sequence_list, 
								annotation_types, 
								max_construction_length, 
								frequency_threshold_constructions_perfile
								):
	
	import pandas as pd
	import cytoolz as ct
	import time
	from candidate_extraction.find_template_matches import find_template_matches
	from candidate_extraction.get_template_name import get_template_name

	#print("Beginning candidate extraction.")
	start_all = time.time()

	candidate_dictionary = {}
	match_counter = 0
	
	#print("Opened: " + filename + ", Length: " + str(len(current_df)))
	
	#Begin loop through templates#
	for template in sequence_list:
	
		start_file = time.time()
		
		copy_df = current_df.copy("Deep")
		
		template = list(template)
		template_name = get_template_name(template)
		
		candidate_dictionary[str(template_name)] = find_template_matches(copy_df, template_name, frequency_threshold_constructions_perfile)
		
		match_counter += len(candidate_dictionary[str(template_name)])
		print("Total: " + str(match_counter) + filename + ": " + str(template_name) + ": "  + str(time.time() - start_file))
			
	#End loop through templates#
	
	end_all = time.time()
	#print("File Time: " + filename + ": " + str(end_all - start_all))
	#print("")
	
	#print("Total candidates: " + str(match_counter))

	return candidate_dictionary
#---------------------------------------------------------------------------------------------#