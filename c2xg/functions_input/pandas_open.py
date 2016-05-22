#---------------------------------------------------------------------------------------------#
#INPUT: List of files containing formatted corpus, file name to write, max sentences ---------#
#OUTPUT: List of sentences contained in input files ------------------------------------------#
#Open files, loop through lines, send lines to other functions, write list of sentences ------#
#---------------------------------------------------------------------------------------------#
def pandas_open(file, 
				encoding_type, 
				semantic_category_dictionary, 
				word_list, 
				lemma_list, 
				pos_list, 
				lemma_dictionary, 
				pos_dictionary, 
				category_dictionary, 
				save_words = False,
				write_output = True,
				delete_temp = False
				):

	import pandas as pd
	import cytoolz as ct
	from functools import partial
	import time
	
	begin = time.time()
	
	print("Ingesting file: " + str(file))
	
	temp_dataframe = pd.read_csv(file, 
									sep="\t", 
									engine="c", 
									header=None, 
									names=['Word', 'Lem', 'Pos', 'Ind', 'Hea', 'Rol'], 
									encoding=encoding_type,
									quotechar="\t",							
									error_bad_lines=False, 
									skip_blank_lines=True
									)
									
	temp_dataframe = temp_dataframe.loc[:,['Ind', 'Word', 'Lem', 'Pos']]	
	#print("Adding semantic category labels")
	
	apply_get = partial(ct.get, seq=semantic_category_dictionary, default="n/a")
	temp_dataframe.loc[:,'Cat'] = temp_dataframe.loc[:,'Word'].str.lower().apply(apply_get)
	
	#print("Indexing Lemma")
	
	apply_get = partial(ct.get, seq=lemma_dictionary, default=0)
	temp_dataframe.loc[:,'Lem'] = temp_dataframe.loc[:,'Lem'].str.lower().apply(apply_get)
	
	#print("Indexing POS")
	
	apply_get = partial(ct.get, seq=pos_dictionary, default=0)
	temp_dataframe.loc[:,'Pos'] = temp_dataframe.loc[:,'Pos'].str.lower().apply(apply_get)
	
	#print("Indexing Category")
	
	apply_get = partial(ct.get, seq=category_dictionary, default=0)
	temp_dataframe.loc[:,'Cat'] = temp_dataframe.loc[:,'Cat'].str.lower().apply(apply_get)
	
	#print("Adding Sentence IDs and removing Sentence markers")
	sentence_list = []
	
	for row in temp_dataframe.itertuples(index=False):

		if pd.notnull(row[1]):
		
			temp_id = row[1]
			
			if "<s:" in temp_id:
				current_id = temp_id.replace("<s:", "").replace(">", "")
				current_id = int(current_id)

		sentence_list.append(current_id)		
	
	temp_dataframe.loc[:,'Sent'] = pd.Series(sentence_list, index=None)
	
	temp_dataframe['Word'] = temp_dataframe['Word'].astype(str)
	temp_dataframe = temp_dataframe[~temp_dataframe['Word'].str.contains("<")]
	
	#print("Removing Word-Forms and Adding Master Index")
	
	sLength = len(temp_dataframe['Ind'])
	temp_dataframe.loc[:,'Mas'] = pd.Series(range(0, sLength), index=temp_dataframe.index)
	
	if save_words == True:
		temp_dataframe = temp_dataframe.loc[:,['Sent', 'Ind', 'Word', 'Lem', 'Pos', 'Cat', 'Mas']]
		
	else:
		temp_dataframe = temp_dataframe.loc[:,['Sent', 'Ind', 'Lem', 'Pos', 'Cat', 'Mas']]
	
	end = time.time()
	print("Ingest Time: " + str(end - begin))
	
	if delete_temp == True:
		import os
		os.remove(file)
	
	if write_output == True:
		
		from functions_input.check_data_files import check_data_files
		from functions_input.get_temp_filename import get_temp_filename
		
		output_file = get_temp_filename(file, ".Pandas")
		check_data_files(output_file)

		temp_dataframe.to_hdf(output_file, "Table", format='table', complevel=9, complib="blosc")

		return output_file
	
	else:

		return temp_dataframe
#---------------------------------------------------------------------------------------------#