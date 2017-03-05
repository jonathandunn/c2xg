#---------------------------------------------------------------------------------------------#
#FUNCTION: get_sentence_length_column --------------------------------------------------------#
#INPUT: Expanded DataFrame -------------------------------------------------------------------#
#OUTPUT: Dataframe with length of each sentence / text in words ------------------------------#
#---------------------------------------------------------------------------------------------#
def get_sentence_length_column(current_df, number_of_sentences):
	
	import pandas as pd
	import numpy as np
	import cytoolz as ct
	import time
	
	start_all = time.time()

	length_list = []
	
	search_df = current_df.loc[:,['Sent', 'Alt', "Lex"]]
	search_df = search_df.query("(Alt == 0)", parser='pandas', engine='numexpr')

	current_sentences = search_df.loc[:,'Sent'].tolist()
	
	frequency_index = ct.frequencies(current_sentences)
	
	for i in range(1, number_of_sentences+1):

		try:
			length_list.append(frequency_index[i])
			
		except:
			length_list.append(0)
	
	temp_series = pd.Series(length_list, name = "Length")
	temp_series.index = np.arange(1, len(temp_series)+1)
	
	end_all = time.time()
	print("Total time for sentence length column: " + str(end_all - start_all))
	
	return temp_series
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#