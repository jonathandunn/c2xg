#---------------------------------------------------------------------------------------------#
#INPUT: DataFrame, index lists ---------------------------------------------------------------#
#OUTPUT: Dictionary of frequency dictionaries ------------------------------------------------#
#Count individual items in unexpanded DataFrame ----------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_frequencies(current_df, Grammar):

	import pandas as pd
	import time
	
	start = time.time()
	
	lemma_frequency = {}
	pos_frequency = {}
	category_frequency = {}
	
	count = current_df.loc[:,"Lex"].value_counts(sort=True, ascending=True, dropna=True)
	for row in count.iteritems():
		unit = row[0]
		label = Grammar.Lemma_List[unit]
		count = row[1]
		lemma_frequency[label] = count
		
	count = current_df.loc[:,'Pos'].value_counts(sort=True, ascending=True, dropna=True)
	for row in count.iteritems():
		unit = row[0]
		label = Grammar.POS_List[unit]
		count = row[1]
		pos_frequency[label] = count
		
	count = current_df.loc[:,'Cat'].value_counts(sort=True, ascending=True, dropna=True)
	for row in count.iteritems():
		unit = row[0]
		label = Grammar.Category_List[unit]
		count = row[1]
		category_frequency[label] = count
		
	end = time.time()
	print("Create frequency dictionaries: " + str(end-start))
	
	current_df = current_df.query("(Pos != 0)")
	number_of_words = len(current_df)
			
	return lemma_frequency, pos_frequency, category_frequency, number_of_words
#---------------------------------------------------------------------------------------------#