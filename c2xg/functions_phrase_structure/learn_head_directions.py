#---------------------------------------------------------------------------------------------#
#INPUT: Data files, current pos tag, and ngram parameters ------------------------------------#
#OUTPUT: Pos tag and head status: Non-Head, Head-First, Head-Last ----------------------------#
#Take data files and pos tags and return head status for each tag ----------------------------#
#---------------------------------------------------------------------------------------------#
def learn_head_directions(pos_label, 
							data_files, 
							phrase_structure_ngram_length, 
							index_list, 
							significance
							):

	import pandas as pd
	import cytoolz as ct
	import statistics
	from scipy import stats
	import decimal as dc
	
	from functions_phrase_structure.get_pos_ngrams import get_pos_ngrams
	
	pos_index = index_list.index(pos_label)
	
	null_counter = 0
	
	#Create listof n-gram count dictionaries from all files#
	
	full_lr_list = []
	full_rl_list = []
	
	for i in range(0,phrase_structure_ngram_length-1):
		temp_dictionary= {}
		full_lr_list.append(temp_dictionary)
		full_rl_list.append(temp_dictionary)
	
	for file in data_files:
		
		current_df = pd.read_hdf(file, key="Table")
		
		current_df.reset_index(inplace=True)
		current_df = current_df.loc[:,['Sent','Pos']]
		
		copy_df = current_df.copy("Deep")
		current_lr_list = get_pos_ngrams(pos_index, phrase_structure_ngram_length, copy_df, "LR")
		
		copy_df = current_df.copy("Deep")
		current_rl_list = get_pos_ngrams(pos_index, phrase_structure_ngram_length, copy_df, "RL")
		
		for i in range(0,phrase_structure_ngram_length-1):
		
			temp_lr_dictionary = ct.merge_with(sum, current_lr_list[i], full_lr_list[i])
			temp_rl_dictionary = ct.merge_with(sum, current_rl_list[i], full_rl_list[i])
			
			try:
				del temp_lr_dictionary['']
				del temp_rl_dictionary['']
			
			except:
				null_counter += 1
			
			full_lr_list[i] = temp_lr_dictionary
			full_rl_list[i] = temp_rl_dictionary
		
	#Done creating n-gram count dictionaries#
	#For each n-gram window, check for significant difference in mean frequency#
	#If there is a significant difference, highest mean is head-side#
	#All identifications must agree#
	lr_counter = 0
	rl_counter = 0
	
	for i in range(len(full_lr_list)):
	
		if len(full_lr_list[i]) > 2 and len(full_rl_list[i]) > 2:
			temp_dictionary = full_lr_list[i]
			lr_types = len(temp_dictionary)
			lr_tokens = sum(temp_dictionary.values())
			lr_mean = dc.Decimal(statistics.mean(temp_dictionary.values()))
		
			temp_dictionary = full_rl_list[i]
			rl_types = len(temp_dictionary)
			rl_tokens = sum(temp_dictionary.values())
			rl_mean = dc.Decimal(statistics.mean(temp_dictionary.values()))
		
			lr_list = list(full_lr_list[i].values())
			rl_list = list(full_rl_list[i].values())
		
			try:
				p_value = stats.ttest_ind(lr_list,rl_list, equal_var = False)
				p_value = dc.Decimal(p_value[1])
			
			except:
				p_value = 1
			
			try:
				if p_value < significance:
		
					if lr_mean > rl_mean:
						if lr_types < rl_types:
							lr_counter += 1
				
					elif rl_mean > lr_mean:
						if rl_types < lr_types:
							rl_counter += 1
			
			except:
				status = "Non-Head"
			
		else:
			status = "Non-Head"
				
	#Check counters to make prediction#
	if lr_counter > 0 and rl_counter == 0:
		status = "Head-First"
		
	elif rl_counter > 0 and lr_counter == 0:
		status = "Head-Last"
		
	else:
		status = "Non-Head"
		
	print("\t" + pos_label + ": " + status)
		
	return (pos_label, status)
#---------------------------------------------------------------------------------------------#