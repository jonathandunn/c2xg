#---------------------------------------------------------------------------------------------#
#INPUT: Sentence and list of reductions ------------------------------------------------------#
#OUTPUT: List of reduced sentence dictionaries -----------------------------------------------#
#Take sentence and list of reductions and return reduced alternate sentences, with ids--------#
#---------------------------------------------------------------------------------------------#
def create_alternate_sentences(single_df, alt_list):

	from functions_constituent_reduction.remove_punc import remove_punc
	
	import pandas as pd
	import time
	
	print("")
	print("Starting to reduce and reform alternate sentence DataFrame.")
	
	#Remove illegal POS values (e.g., punctuation)#
	print("\tRemoving illegal POS")
	alt_list.append(remove_punc(single_df, 0))
		
	temp_dataframe = pd.concat(alt_list)
	temp_dataframe = temp_dataframe.sort_values(by=['Sent', 'Alt', 'Mas'], axis=0, ascending=True, inplace=False, kind='mergesort')
	temp_dataframe = temp_dataframe.loc[:,['Sent', 'Alt', 'Mas', 'Lem', 'Pos', 'Cat']]
	#Finished creating and formatting DataFrame#
	
	return temp_dataframe
#---------------------------------------------------------------------------------------------#