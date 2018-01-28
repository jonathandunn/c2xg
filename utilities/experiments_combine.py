#---------------------------------------------------------------------------------------------#
#FUNCTION: read_candidates -------------------------------------------------------------------#
#INPUT: Name for candidate H5 file -----------------------------------------------------------#
#OUTPUT: List of candidates ------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def read_candidates(file):
    
	import pickle
	
	candidate_list = []
	
	with open(file,'rb') as f:
		candidate_list = pickle.load(f)
		
	return candidate_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------#
#FUNCTION: write_candidates ------------------------------------------------------------------#
#INPUT: Name for candidate file  and list of candidates---------------------------------------#
#OUTPUT: None: Write file with candidates ----------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def write_candidates(file, candidate_list):
    
	import pickle
	import os.path
	import os
	
	if os.path.isfile(file):
		os.remove(file)
		
	try:
	
		with open(file,'wb') as f:
			pickle.dump(candidate_list,f)
			
	except:
	
		print("Problem with " + file)
	
	return
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------#
#FUNCTION: merge_output ----------------------------------------------------------------------#
#INPUT: List of output files from Learning.2.Processing.py -----------------------------------#
#OUTPUT: Dictionary with all acceptable candidates, their frequency, other info --------------#
#---------------------------------------------------------------------------------------------#
def merge_output(output_files):

	import cytoolz as ct
	
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
			
	print("")
	print("Done merging.")
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
	final_dictionary['candidate_dictionary'] = candidate_dictionary
	
	del candidate_dictionary
	
	return final_dictionary
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
# filename = "./candidates_combined/Candidates.Merged.001-010.p"

# output_files = [
# "./candidates/Candidates.UKWAC-1.5k.001.txt.Reformatted.txt.p",
# "./candidates/Candidates.UKWAC-1.5k.002.txt.Reformatted.txt.p",
# "./candidates/Candidates.UKWAC-1.5k.003.txt.Reformatted.txt.p",
# "./candidates/Candidates.UKWAC-1.5k.004.txt.Reformatted.txt.p",
# "./candidates/Candidates.UKWAC-1.5k.005.txt.Reformatted.txt.p",
# "./candidates/Candidates.UKWAC-1.5k.006.txt.Reformatted.txt.p",
# "./candidates/Candidates.UKWAC-1.5k.007.txt.Reformatted.txt.p",
# "./candidates/Candidates.UKWAC-1.5k.008.txt.Reformatted.txt.p",
# "./candidates/Candidates.UKWAC-1.5k.009.txt.Reformatted.txt.p",
# "./candidates/Candidates.UKWAC-1.5k.010.txt.Reformatted.txt.p"
# ]

# final_dictionary = merge_output(output_files)
# write_candidates(filename, final_dictionary)