#---------------------------------------------------------------------------------------------#
#INPUT: Line ---------------------------------------------------------------------------------#
#OUTPUT: Alternate sentences -----------------------------------------------------------------#
#Allows parrallel processing of alternate candidate generation -------------------------------#
#---------------------------------------------------------------------------------------------#
def process_sentence_expansion(current_df, 
								lemma_list, 
								pos_list, 
								category_list, 
								phrase_constituent_list
								):

	from functions_constituent_reduction.process_learned_constituents import process_learned_constituents
	from functions_constituent_reduction.process_schematic_representation import process_schematic_representation
	from functions_constituent_reduction.create_alternate_sentences import create_alternate_sentences
	
	import pandas as pd
	import time
	
	single_df = current_df.copy("Deep")
	
	single_df.reset_index(inplace=True)	
	single_df = single_df.loc[:,['Sent', 'Mas', 'Lem', 'Pos', 'Cat']]
	
	alt_list = []
	
	#Counter keeps track of constituent types for assigning "Alt" ids#
	#Alt == 0 is reserved for the original input#
	
	#Remove Head-First constituents#
	print("")
	print("\tStarting Head-First Constituent Reduction")
	start = time.time()
	total_match_df_lr, remove_dictionary_lr, counter = process_learned_constituents(single_df, 
																					pos_list, 
																					lemma_list, 
																					phrase_constituent_list[0], 
																					direction = "LR", 
																					action = "Reduce", 
																					counter = 1
																					)
	alt_list.append(total_match_df_lr)

	end = time.time()		
	print("\tDone with Head-First phrases: " + str(end - start) + ", Number of alts: " + str(counter))
	
	#Remove Head-Last constituents#
	print("")
	print("\tStarting Head-Last Constituent Reduction")
	start = time.time()
	total_match_df_rl, remove_dictionary_rl, counter = process_learned_constituents(single_df, 
																					pos_list, 
																					lemma_list, 
																					phrase_constituent_list[1], 
																					direction = "RL", 
																					action = "Reduce", 
																					counter = counter
																					)
	alt_list.append(total_match_df_rl)

	end = time.time()		
	print("\tDone with Head-Last phrases: " + str(end - start) + ", Number of alts: " + str(counter))
											
	#Fully schematic representation#
	print("")
	print("\tStarting Fully Schematic Representation")
	start = time.time()
	total_schematic_df, counter = process_schematic_representation(single_df, 
															pos_list, 
															lemma_list, 
															remove_dictionary_lr, 
															remove_dictionary_rl, 
															counter
															)
	alt_list.append(total_schematic_df)

	end = time.time()
	print("\tDone with Fully-Schematic Representation: " + str(end - start) + ", Number of alts: " + str(counter))
	
	#Call function to combine and reformat alternate sentence DataFrames#
	alternate_sentence_candidates = create_alternate_sentences(single_df, alt_list)

	return alternate_sentence_candidates
#---------------------------------------------------------------------------------------------#