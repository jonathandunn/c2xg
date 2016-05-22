#---------------------------------------------------------------------------------------------#
#INPUT: Data files with list of sentence dictionaries ----------------------------------------#
#OUTPUT: Number of expanded sentences; list of dictionaries saved to file --------------------#
#Take sentences and generate simpler, non-recursive variations and save them to file ---------#
#---------------------------------------------------------------------------------------------#
def expand_sentences(data_files, 
						lemma_list, 
						pos_list, 
						category_list, 
						encoding_type, 
						write_output, 
						phrase_constituent_list
						):

	import pandas as pd
	from functions_constituent_reduction.process_sentence_expansion import process_sentence_expansion
	
	if write_output == True:
		print("Start sentence expansion for " + str(data_files))
	
	#Open HDF5 data file#
	if write_output == True:
		data_file_expanded = data_files + ".Expanded"
		store = pd.HDFStore(data_files)
		current_df = store['Table']
		store.close()
		
	elif write_output == False:
		current_df = data_files
	
	temp_dataframe = process_sentence_expansion(current_df, lemma_list, pos_list, category_list, phrase_constituent_list)
	
	if write_output == True:
		
		print("Done expanding sentences for " + str(data_files) + ", Lines: " + str(len(temp_dataframe)))
		#Save expanded sentence representation for expanded datafile#
		temp_dataframe.to_hdf(data_file_expanded, "Table", format="table", complevel=9, complib="blosc")

		return
		
	elif write_output == False:
		
		return temp_dataframe
#---------------------------------------------------------------------------------------------#