#---------------------------------------------------------------------------------------------#
#INPUT: Data files, current head, and ngram parameters ---------------------------------------#
#OUTPUT: Tuple with phrase type and constituent instances for the phrase ---------------------#
#Take data files and phrase type, extract ngrams and prune to valid constituents -------------#
#---------------------------------------------------------------------------------------------#
def learn_phrase_constituents(pos_tuple, 
								data_files, 
								phrase_structure_ngram_length, 
								index_list, 
								lr_head_list, 
								rl_head_list, 
								constituent_threshold
								):

	import pandas as pd
	import cytoolz as ct
	
	from functions_phrase_structure.get_pos_ngrams import get_pos_ngrams
	from functions_phrase_structure.prune_constituents import prune_constituents
	
	pos_label = pos_tuple[0]
	pos_index = index_list.index(pos_label)
	pos_direction = pos_tuple[1]
	
	null_counter = 0
	
	if pos_direction != "Non-Head":
	
		if pos_direction == "Head-First":
			direction = "LR"
			
		elif pos_direction == "Head-Last":
			direction = "RL"
	
		#Create listof n-gram count dictionaries from all files#
	
		full_list = []
	
		for i in range(0,phrase_structure_ngram_length-1):
			temp_dictionary = {}
			full_list.append(temp_dictionary)
	
		for file in data_files:
		
			current_df = pd.read_hdf(file, key="Table")
		
			current_df.reset_index(inplace=True)
			current_df = current_df.loc[:,['Sent','Pos']]
		
			copy_df = current_df.copy("Deep")
			current_list = get_pos_ngrams(pos_index, phrase_structure_ngram_length, copy_df, direction)
			
			for i in range(0,phrase_structure_ngram_length-1):
		
				temp_dictionary = ct.merge_with(sum, current_list[i], full_list[i])
				
				try:
					del temp_dictionary['']
				except:
					null_counter += 1
			
				full_list[i] = temp_dictionary
				
		ngram_dictionary = ct.merge([x for x in full_list])
		
		ngram_dictionary = prune_constituents(ngram_dictionary, 
												pos_direction, 
												index_list, 
												lr_head_list, 
												rl_head_list, 
												pos_index, 
												constituent_threshold
												)
		
		print("\tDone with " + str(pos_label) + ", " + pos_direction + ": " + str(len(ngram_dictionary.keys())))
	
		return (pos_index, list(ngram_dictionary.keys()))
		
	else:
	
		print("\tDone with " + str(pos_label) + ", a Non-Head")
		
		return
#---------------------------------------------------------------------------------------------#