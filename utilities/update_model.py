#---------------------------------------------------------------------------------------#
#---Short script for removing infrequent units and rewriting usage model----------------#
#---------------------------------------------------------------------------------------#
def update_model(model_file):

	from functions_candidate_extraction.read_candidates import read_candidates
	from functions_candidate_extraction.write_candidates import write_candidates

	write_dictionary = read_candidates(model_file)

	lemma_list = write_dictionary['lemma_list']
	centroid_df = write_dictionary['centroid_df']
	
	frequency_list = []
	
	for word in lemma_list:
		
		new_word = "Lem:" + word + " "
		new_word = new_word.encode()

		try:
			frequency = float(centroid_df[new_word].values)
			frequency_list.append(frequency)
			
		except:
			print("!!!!!Doesn't exist: " + str(new_word))
			
	mean_frequency = sum(frequency_list) / float(len(frequency_list))
	print("Mean Frequency: " + str(mean_frequency))
	
	for word in lemma_list:
		
		new_word = "Lem:" + word + " "
		new_word = new_word.encode()

		try:
			frequency = float(centroid_df[new_word].values)
			
			if frequency < mean_frequency:

				centroid_df.drop(new_word, axis=1, inplace=True, errors='raise')
				lemma_list.remove(word)
			
		except:
			print("!!!!!Error: " + str(new_word))
		
	
	write_dictionary['lemma_list'] = lemma_list
	write_dictionary['centroid_df'] = centroid_df
	
	write_candidates(model_file, write_dictionary)	
	
	
	return
#------------------------------------------------------------------------------------#

model_file = "E:/data/4.Output/English.Twitter.Full.4-8.3.Usage.model"
update_model(model_file)