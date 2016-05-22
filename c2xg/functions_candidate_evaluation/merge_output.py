#---------------------------------------------------------------------------------------------#
#FUNCTION: merge_output ----------------------------------------------------------------------#
#INPUT: List of output files from Learning.2.Processing.py -----------------------------------#
#OUTPUT: Dictionary with all acceptable candidates, their frequency, other info --------------#
#---------------------------------------------------------------------------------------------#
def merge_output(output_files, frequency_threshold):

	from functions_candidate_extraction.read_candidates import read_candidates
	import cytoolz as ct
	
	temp_dir_index = output_files[0].rfind("/")
	temp_dir = output_files[0][temp_dir_index:]
	
	lemma_frequency = {}
	pos_frequency = {}
	category_frequency = {}
	number_of_words_total = 0
	candidate_dictionary = {}
	dictionary_key_list = []
	
	final_dictionary = {}
	
	print("")
	print("Merging candidate output files from processing stage.")

	#First, read in one output file to create the baseline items#
	print("\tStarting with: " + str(output_files[0]))
	current_dictionary = read_candidates(output_files[0])
		
	#These items are shared and need to be saved once and checked for consistency only#
	pos_list = current_dictionary['pos_list']
	lemma_list = current_dictionary['lemma_list']
	category_list = current_dictionary['category_list']
	word_list = current_dictionary['word_list']
	phrase_constituent_list = current_dictionary['phrase_constituent_list']
	semantic_category_dictionary = current_dictionary['semantic_category_dictionary']
	lemma_dictionary = current_dictionary['lemma_dictionary']
	pos_dictionary = current_dictionary['pos_dictionary']
	category_dictionary = current_dictionary['category_dictionary']
	emoji_dictionary = current_dictionary['emoji_dictionary']
				
	#These items need to be merged across all output files#
	lemma_frequency = current_dictionary['lemma_frequency']
	pos_frequency = current_dictionary['pos_frequency']
	category_frequency = current_dictionary['category_frequency']
	number_of_words_total += current_dictionary['number_of_words']	
	candidate_dictionary = current_dictionary['candidate_dictionary']
	dictionary_key_list = list(current_dictionary['candidate_dictionary'].keys())
	
	del current_dictionary
	
	#Second, for all other files, make sure they have same info and merge results#
	for file in output_files[1:]:
	
		print("\tAdding: " + str(file))
		
		current_dictionary = {}
		current_dictionary = read_candidates(file)
		
		temp_dictionary = {}
		
		#Check to ensure the otuput files were created with the same grammar file#
		if current_dictionary['pos_list'] == pos_list and current_dictionary['phrase_constituent_list'] == phrase_constituent_list:
		
			#These items need to be merged across all output files#
			lemma_frequency = ct.merge_with(sum, [lemma_frequency, current_dictionary['lemma_frequency']])
			pos_frequency = ct.merge_with(sum, [pos_frequency, current_dictionary['pos_frequency']])
			category_frequency = ct.merge_with(sum, [category_frequency, current_dictionary['category_frequency']])
			
			number_of_words_total += current_dictionary['number_of_words']	
			
			dictionary_key_list += list(current_dictionary['candidate_dictionary'].keys())
			dictionary_key_list = list(set(dictionary_key_list))

			for key in dictionary_key_list:
				temp_dictionary[key] = ct.merge_with(sum, [candidate_dictionary[key], current_dictionary['candidate_dictionary'][key]])
				
			del candidate_dictionary
			candidate_dictionary = temp_dictionary
			del temp_dictionary
			
		else:
			print("\t\tFile " + str(file) + " did not match grammar elements. Not compatible.")
			
	#Count total candidates#
	total = 0
	for key in candidate_dictionary.keys():
		total += len(candidate_dictionary[key].keys())
	#Done counting#
	
	print("")
	print("Done merging frequency dictionaries. Now pruning the " + str(total) + " candidates.")
	print("\tFrequency threshold is " + str(frequency_threshold) + " occurrences.")
	check_threshold = lambda x: x > frequency_threshold
	final_candidate_dictionary = {}
	
	for key in candidate_dictionary.keys():
		final_candidate_dictionary[key] = ct.valfilter(check_threshold, candidate_dictionary[key])
		
	#Count total candidates#
	total = 0
	for key in final_candidate_dictionary.keys():
		total += len(final_candidate_dictionary[key].keys())
	#Done counting#
	
	print("")
	print("Done merging and reducing data: " + str(total) + " candidates after frequency threshold.")
	print("")
	
	final_dictionary['pos_list'] = pos_list
	final_dictionary['lemma_list'] = lemma_list
	final_dictionary['category_list'] = category_list
	final_dictionary['word_list'] = word_list
	final_dictionary['phrase_constituent_list'] = phrase_constituent_list
	final_dictionary['semantic_category_dictionary'] = semantic_category_dictionary
	final_dictionary['lemma_dictionary'] = lemma_dictionary
	final_dictionary['pos_dictionary'] = pos_dictionary
	final_dictionary['category_dictionary'] = category_dictionary
	final_dictionary['emoji_dictionary'] = emoji_dictionary
	
	final_dictionary['lemma_frequency'] = lemma_frequency
	final_dictionary['pos_frequency'] = pos_frequency
	final_dictionary['category_frequency'] = category_frequency
	final_dictionary['number_of_words'] = number_of_words_total
	final_dictionary['candidate_dictionary'] = final_candidate_dictionary
	
	return final_dictionary
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#