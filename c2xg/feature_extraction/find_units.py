#---------------------------------------------------------------------------------------------#
#INPUT: Current DataFrame and index lists ----------------------------------------------------#
#OUTPUT: DataFrame with matches to template --------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def find_units(current_df, unit_list, unit_type, number_of_sentences, lemma_list, pos_list, category_list):
    
	import pandas as pd
	import cytoolz as ct
	import numpy as np
	import time
	
	from feature_extraction.get_vector_column import get_vector_column
	from feature_extraction.get_construction_name import get_construction_name
	
	start_all = time.time()	
	
	vector_column_list = []
	column_names = []
	
	#Make zero list for non-matches#
	zero_list = []
	for i in range(1,number_of_sentences+1):
		zero_list.append(0)
	
	search_df = current_df.loc[:,['Sent', 'Alt', str(unit_type)]]
	query_string = "(Alt == 0)"
	search_df = search_df.query(query_string, parser='pandas', engine='numexpr')
	
	for i in range(1,len(unit_list)):
	
		#Check to prevent lemma phrases from being counted#
		if unit_type == "Lex" and "_PHRASE" in unit_list[i]:
			print("", end="")
		
		#If not a lemma phrase, proceed as planned#
		else:
			candidate = [(unit_type, i)]
		
			query_string = "(" + unit_type + " == " + str(i) + ")"
			match_df = search_df.query(query_string, parser='pandas', engine='numexpr')
		
			#Check to prevent counting of non-existent features#
			if len(match_df) > 0:
		
				current_sentences = []
			
				#Create vector for current construction#
				current_sentences = match_df.loc[:,'Sent'].tolist()
				candidate_list = get_vector_column(current_sentences, number_of_sentences)
				candidate_id = get_construction_name(candidate, lemma_list, pos_list, category_list)
				temp_series = pd.Series(candidate_list, name = candidate_id)
				temp_series.index = np.arange(1, len(temp_series)+1)
				vector_column_list.append(temp_series)
				
				del temp_series
				del candidate_list
					
			#Process non-existent features#
			else:
			
				candidate_id = get_construction_name(candidate, lemma_list, pos_list, category_list)
				series_list = zero_list
				
				temp_series = pd.Series(series_list, name = candidate_id)
				temp_series.index = np.arange(1, len(temp_series)+1)
				vector_column_list.append(temp_series)

				del series_list
				del temp_series
			
			del match_df
			#Done with feature match check#
		#Done with lemma phrase check#	
		
	#Done counting features. Now create DataFrame for results#
	results_df = pd.concat(vector_column_list, axis = 1)
	
	end_all = time.time()
	print("Total time for extraction of " + str(unit_type) + ": " + str(end_all - start_all))
	
	return results_df
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#