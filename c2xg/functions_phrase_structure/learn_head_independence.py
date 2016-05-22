#---------------------------------------------------------------------------------------------#
#INPUT: Data files, current pos tag, and ngram parameters ------------------------------------#
#OUTPUT: Pos tag and independence status: Can a single head constitute a phrase? -------------#
#Take data files and pos tags and return independence status for each tag --------------------#
#---------------------------------------------------------------------------------------------#
def learn_head_independence(pos_label, 
								data_files, 
								index_list, 
								lr_head_list, 
								rl_head_list,
								independence_threshold
								):

	import pandas as pd
	import cytoolz as ct
	from functions_phrase_structure.get_pos_ngrams import get_pos_ngrams
	
	pos_index = index_list.index(pos_label)
	
	#Find head-status of current unit#
	if pos_index in lr_head_list:
		current_direction = "LR"
	
	elif pos_index in rl_head_list:
		current_direction = "RL"
		
	else:
		current_direction = "None"
	
	#Create list of pos bi-grams starting with current head and going in head direction#
	if current_direction != "None":
		
		full_dictionary = {}
	
		for file in data_files:
		
			current_df = pd.read_hdf(file, key="Table")
		
			current_df.reset_index(inplace=True)
			current_df = current_df.loc[:,['Sent','Pos']]
		
			copy_df = current_df.copy("Deep")
			current_dictionary = get_pos_ngrams(pos_index, 2, copy_df, current_direction)
			current_dictionary = current_dictionary[0]
			
			try:
				del current_dictionary[""]
			except:
				null_counter = 1
						
			full_dictionary = ct.merge_with(sum, current_dictionary, full_dictionary)
	
		#Done looping through files. Now create frequency dictionary.#
		current_tuple = str("(" + str(pos_index) + ", " + str(pos_index) + ")")
		
		try:
			current_adjacent = full_dictionary[current_tuple]
			
		except:
			current_adjacent = 0
	
		if independence_threshold < current_adjacent:
			status = "Independent"
		
		else:
			status = "Dependent"

		print("\t" + pos_label + ": " + status)
		return (pos_label, status)
	
	#If current unit is not a head at all, return nothing#
	else:
		status = "Non-Head"
		print("\t" + pos_label + ": " + "is not a head.")
		return (pos_label, status)
#---------------------------------------------------------------------------------------------#