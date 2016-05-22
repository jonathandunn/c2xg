#---------------------------------------------------------------------------------------------#
#INPUT: DataFrame, index lists ---------------------------------------------------------------#
#OUTPUT: Dictionary of frequency dictionaries ------------------------------------------------#
#Count individual items in unexpanded DataFrame ----------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_frequencies(current_df, 
					lemma_list, 
					pos_list, 
					category_list
					):

	import pandas as pd
	import time
	
	start = time.time()
	
	freq_dict = {}
	lemma_frequency = {}
	pos_frequency = {}
	category_frequency = {}
	
	count = current_df.loc[:,'Lem'].value_counts(sort=True, ascending=True, dropna=True)
	for row in count.iteritems():
		unit = row[0]
		label = lemma_list[unit]
		count = row[1]
		lemma_frequency[label] = count
		
	count = current_df.loc[:,'Pos'].value_counts(sort=True, ascending=True, dropna=True)
	for row in count.iteritems():
		unit = row[0]
		label = pos_list[unit]
		count = row[1]
		pos_frequency[label] = count
		
	count = current_df.loc[:,'Cat'].value_counts(sort=True, ascending=True, dropna=True)
	for row in count.iteritems():
		unit = row[0]
		label = category_list[unit]
		count = row[1]
		category_frequency[label] = count
		
	end = time.time()
	print("Create frequency dictionaries: " + str(end-start))
	
	freq_dict['lemma_frequency'] = lemma_frequency
	freq_dict['pos_frequency'] = pos_frequency
	freq_dict['category_frequency'] = category_frequency

	current_df = current_df.query("(Pos != 0)")
	number_of_words = len(current_df)
	
	freq_dict['number_of_words'] = number_of_words
			
	return freq_dict
#---------------------------------------------------------------------------------------------#