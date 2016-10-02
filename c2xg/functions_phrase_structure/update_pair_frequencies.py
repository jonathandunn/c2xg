#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def update_pair_frequencies(pos_index, 
							index_list, 
							current_df,
							base_frequency_dictionary,
							lemma_list
							):

	import cytoolz as ct
	from functions_phrase_structure.get_pos_ngrams import get_pos_ngrams
	
	if index_list[pos_index] != "n/a":
		
		
		
		#Get dictionary of n-grams#
		
		copy_df = current_df.copy("Deep")
		ngram_dictionary_lr = get_pos_ngrams(pos_index, 2, copy_df, "LR", lemma_list, index_list)
		
		copy_df = current_df.copy("Deep")
		ngram_dictionary_rl = get_pos_ngrams(pos_index, 2, copy_df, "RL", lemma_list, index_list)
		
		for pair_frequency_dictionary in [ngram_dictionary_lr, ngram_dictionary_rl]:
			for key in pair_frequency_dictionary.keys():
				
				if base_frequency_dictionary[key[0]] < pair_frequency_dictionary[key] or base_frequency_dictionary[key[1]] < pair_frequency_dictionary[key]:
					print(pair_frequency_dictionary[key], end="")
					print(" : " + str(key) + " : ", end="")
					print(base_frequency_dictionary[key[0]], end="")
					print(" and ", end="")
					print(base_frequency_dictionary[key[1]])
				
		pair_frequency_dictionary = ct.merge(ngram_dictionary_rl, ngram_dictionary_lr)
		
		return pair_frequency_dictionary
		
	else:
	
		return
#---------------------------------------------------------------------------------------------#