#---------------------------------------------------------------------------------------------#
#FUNCTION: write_model -----------------------------------------------------------------------#
#INPUT: Data required for autonomous feature extraction --------------------------------------#
#OUTPUT: Model file for feature extraction with model file and new text ----------------------#
#---------------------------------------------------------------------------------------------#
def write_model(lemma_list, 
				pos_list, 
				word_list, 
				category_list, 
				semantic_category_dictionary, 
				sequence_list, 
				max_construction_length, 
				annotation_types, 
				candidate_list, 
				encoding_type, 
				data_file_model, 
				phrase_constituent_list, 
				lemma_dictionary, 
				pos_dictionary, 
				category_dictionary, 
				emoji_dictionary
				):
	
	import pandas as pd
	from functions_candidate_extraction.write_candidates import write_candidates
	from functions_candidate_pruning.add_constituent_candidates import add_constituent_candidates
	
	print("Writing model file for autonomous feature extraction.")
	
	write_dictionary = {}
	
	write_dictionary['candidate_list'] = candidate_list
	write_dictionary['lemma_list'] = lemma_list
	write_dictionary['pos_list'] = pos_list
	write_dictionary['word_list'] = word_list
	write_dictionary['category_list'] = category_list
	write_dictionary['semantic_category_dictionary'] = semantic_category_dictionary
	write_dictionary['sequence_list'] = sequence_list
	write_dictionary['max_construction_length'] = max_construction_length
	write_dictionary['annotation_types'] = annotation_types
	write_dictionary['encoding_type'] = encoding_type
	write_dictionary['phrase_constituent_list'] = phrase_constituent_list
	write_dictionary['lemma_dictionary'] = lemma_dictionary
	write_dictionary['pos_dictionary'] = pos_dictionary
	write_dictionary['category_dictionary'] = category_dictionary
	write_dictionary['emoji_dictionary'] = emoji_dictionary
	
	write_candidates(data_file_model, write_dictionary)	
	
	return
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#