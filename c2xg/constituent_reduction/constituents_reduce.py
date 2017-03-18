#---------------------------------------------------------------------------------------------#
#INPUT: Direction, dictionary of longest constituents, DF, and current constituent -----------#
#OUTPUT: Combined DF of alts -----------------------------------------------------------------#
#Reduce constituents in current direction ----------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def constituents_reduce(pos_list, 
						lemma_list, 
						direction, 
						remove_dictionary, 
						copy_df, 
						key, 
						action = "REDUCE", 
						encoding_type = "", 
						examples_file = ""
						):

	import pandas as pd
	from constituent_reduction.get_head_first_list import get_head_first_list
	from constituent_reduction.get_head_last_list import get_head_last_list	
	from constituent_reduction.constituents_print import constituents_print
	
	if direction == "LR":
		remove_list, head_list = get_head_first_list(remove_dictionary)
		
	elif direction == "RL":
		remove_list, head_list = get_head_last_list(remove_dictionary)
	
	#Get list of all sentence involved#
	sentence_df = copy_df.loc[copy_df.Mas.isin(head_list), 'Sent']
	sentence_list = sentence_df.drop_duplicates().values.tolist()
	match_df = copy_df.loc[copy_df.Sent.isin(sentence_list)]
	
	#Remove non-head indexes#
	match_df = match_df[~match_df.Mas.isin(remove_list)]
						
	#Replace head indexes with current phrase type#
	#Replacement depends on head independence status#
	
	#Independent heads: labelled with head part-of-speech, lemma stays the same#
	#Dependent heads: labelled with pos_phrase, lemma changes to pos_phrase#
	
	current_pos_index = key
	match_df.loc[match_df.Mas.isin(head_list), 'Pos'] = current_pos_index
			
	if action == "PRINT":

		original_df = copy_df.loc[copy_df.Sent.isin(sentence_list)]
		constituents_print(pos_list[key], head_list, remove_list, lemma_list, original_df, match_df, direction, examples_file, encoding_type)
			
	print("\t\t" + str(pos_list[key]) + ": " + str(len(head_list)) + " matches.")
			
	return match_df
#---------------------------------------------------------------------------------------------#