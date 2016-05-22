#---------------------------------------------------------------------------------------------#
#INPUT: Template with numbers for column labels ----------------------------------------------#
#OUTPUT: List of template units with strings as column labels --------------------------------#
#---------------------------------------------------------------------------------------------#
def merge_file_results(data_file_candidate_constructions, 
						data_files_expanded, sequence_list, 
						frequency_threshold_constructions
						):
	
	import cytoolz as ct
	import time
	from functions_candidate_extraction.get_candidate_name import get_candidate_name
	from functions_candidate_extraction.read_candidates import read_candidates
	
	full_candidate_list = []
	
	for i in range(len(sequence_list)):
		
		start_file = time.time()
		full_candidate_dictionary = {}
		
		for file in data_files_expanded:
		
			pickled_list_file = get_candidate_name(data_files_expanded, data_file_candidate_constructions)
			pickled_list_file = pickled_list_file.replace("']","")
			current_file_list = read_candidates(pickled_list_file)
			
			current_template = current_file_list[i][0]
			current_file_candidate_dictionary = current_file_list[i][1]
		
			#Merge current candidates with candidates from previous files, summing individual counts#
			full_candidate_dictionary = ct.merge_with(sum, full_candidate_dictionary, current_file_candidate_dictionary)
			
		#Done with all files for this template, now remove infrequent candidates#
		print("Finished candidate merging. Now removing infrequent candidates.")
		above_threshold = lambda x: x > frequency_threshold_constructions
		full_candidate_dictionary = ct.valfilter(above_threshold, full_candidate_dictionary)
		
		#Now add frequency reduced template candidates to full list#
		current_addition = [current_template, full_candidate_dictionary]
		full_candidate_list.append(current_addition)
				
		end_file = time.time()
		print(str(current_template) + ": "  + str(end_file - start_file))
		print("")

	return full_candidate_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#