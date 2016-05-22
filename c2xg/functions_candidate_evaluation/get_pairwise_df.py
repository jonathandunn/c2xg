#---------------------------------------------------------------------------------------------#
#FUNCTION: get_pairwise_df ------------------------------------------------------------------#
#INPUT: Dictionary of pairs and their co-occurrences, unit index and frequency lists ---------#
#OUTPUT: Dataframe with co-occurrence data (a, b, c) for each pair ---------------------------#
#---------------------------------------------------------------------------------------------#
def get_pairwise_df(pairwise_candidate, 
						lemma_frequency, 
						lemma_list, 
						pos_frequency, 
						pos_list, 
						category_frequency, 
						category_list, 
						total_units
						):

	import pandas as pd
	
	candidate = str(pairwise_candidate[0])
	frequency = pairwise_candidate[2]
	
	#Assign label and index values#
	unit1_label = pairwise_candidate[0][0][0]
	unit1_index = pairwise_candidate[0][0][1]
	
	unit2_label = pairwise_candidate[0][1][0]
	unit2_index = pairwise_candidate[0][1][1]
	
	#Choose list for each pair type#
	if unit1_label == "Lem":
		unit1_list = lemma_frequency
		unit1_key = lemma_list[unit1_index]
	
	elif unit1_label == "Pos":
		unit1_list = pos_frequency
		unit1_key = pos_list[unit1_index]
		
	elif unit1_label == "Cat":
		unit1_list = category_frequency
		unit1_key = category_list[unit1_index]
		
	if unit2_label == "Lem":
		unit2_list = lemma_frequency
		unit2_key = lemma_list[unit2_index]
	
	elif unit2_label == "Pos":
		unit2_list = pos_frequency
		unit2_key = pos_list[unit2_index]
		
	elif unit2_label == "Cat":
		unit2_list = category_frequency
		unit2_key = category_list[unit2_index]
	
	#Calculate co-occurence frequencies#
	unit1_freq = unit1_list[unit1_key]
	unit2_freq = unit2_list[unit2_key]
	
	a = frequency
	b = unit1_freq - a
	c = unit2_freq - a
	d = total_units - a - b - c
	
	return [candidate, a, b, c, d]
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#