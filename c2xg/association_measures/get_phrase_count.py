#---------------------------------------------------------------------------------------------#
#FUNCTION: get_phrase_count ------------------------------------------------------------------#
#INPUT: List of unexpanded data files --------------------------------------------------------#
#OUTPUT: Total number of units in corpus -----------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_phrase_count(original_df, Grammar, lemma_frequency, pos_frequency):
	
	import pandas as pd
	
	phrase_names = [x for x in Grammar.POS_List if x not in pos_frequency.keys()]
	
	try:
		del phrase_names[phrase_names.index('n/a')]
	except:
		print("")
	
	#print("")
	#print("Finding frequency for phrase types.")
	
	for key in phrase_names:
		
		pos_frequency[key] = 0
		lemma_frequency[key] = 0	
		
	#Count Phrases#
	for unit in phrase_names:
		
		print("\tCurrent phrase: " + str(unit))
		
		try:
			unit_index = Grammar.Lemma_List.index(unit)
			
		except:
			Grammar.Lemma_List.append(unit)
			unit_index = Grammar.Lemma_List.index(unit)
		
		unit_mask = original_df.loc[:, "Lex"] == unit_index
		unit_df = original_df.loc[unit_mask, ['Sent', 'Unit']]
		unit_df = unit_df.drop_duplicates(keep='first')
		unit_list = unit_df.values.tolist()
			
		current_count = len(unit_list)
		pos_frequency[unit] += current_count
		lemma_frequency[unit] += current_count
			
	return lemma_frequency, pos_frequency
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#