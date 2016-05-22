#---------------------------------------------------------------------------------------------#
#FUNCTION: get_phrase_count ------------------------------------------------------------------#
#INPUT: List of unexpanded data files --------------------------------------------------------#
#OUTPUT: Total number of units in corpus -----------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_phrase_count(lemma_list, 
						lemma_frequency, 
						pos_list, 
						pos_frequency, 
						original_df
						):
	
	import pandas as pd
	
	phrase_names = [x for x in pos_list if x not in pos_frequency.keys()]
	
	try:
		del phrase_names[phrase_names.index('n/a')]
	except:
		print("")
	
	print("")
	print("Finding frequency for phrase types.")
	
	for key in phrase_names:
		
		pos_frequency[key] = 0
		lemma_frequency[key] = 0
	
		
	#Count Phrases#
	for unit in phrase_names:
		
		print("\tCurrent phrase: " + str(unit))
		
		try:
			unit_index = lemma_list.index(unit)
			
		except:
			lemma_list.append(unit)
			unit_index = lemma_list.index(unit)
		
		unit_mask = original_df.loc[:, 'Lem'] == unit_index
		unit_df = original_df.loc[unit_mask, ['Sent', 'Unit']]
		unit_df = unit_df.drop_duplicates(keep='first')
		unit_list = unit_df.values.tolist()
			
		current_count = len(unit_list)
		pos_frequency[unit] += current_count
		lemma_frequency[unit] += current_count
			
	print("")
		
	return_dictionary = {}
	return_dictionary['lemma_frequency'] = lemma_frequency
	return_dictionary['pos_frequency'] = pos_frequency

	return return_dictionary
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#