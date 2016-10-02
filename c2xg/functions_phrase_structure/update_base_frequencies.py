#---------------------------------------------------------------------------------------------#
#INPUT: Frequency dictionary, DataFrame --------------------------------------------#
#OUTPUT: Updated frequency dictionary --------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def update_base_frequencies(base_frequency_dictionary, current_df):

	pos_frequency = current_df.loc[:, "Pos"].value_counts()
	
	for item in pos_frequency.iteritems():
		
		pos_index = item[0]
		pos_count = item[1]
		
		try:
			base_frequency_dictionary[pos_index] += pos_count
			
		except:
			base_frequency_dictionary[pos_index] = pos_count
			
	return base_frequency_dictionary
#---------------------------------------------------------------------------------------------#