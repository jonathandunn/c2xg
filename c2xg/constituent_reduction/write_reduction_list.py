#---------------------------------------------------------------------------------------------#
#INPUT: Name of debug file, single line, alternate sentence representations with debug info --#
#OUTPUT: Add current alternates to readable debug file ---------------------------------------#
#Take line, debug filename, and alternate representations and write readable list of changes--#
#---------------------------------------------------------------------------------------------#
def write_reduction_list(data_files_expanded, 
							word_list, 
							encoding_type
							):

	import pandas as pd
	
	fa = open(data_file_reductions, "a", encoding=encoding_type)
			
	for file in data_files_expanded:
	
		print("Writing readable reductions corpus for: " + str(file))
	
		store = pd.HDFStore(file)
		single_df = store['Table']
		store.close()
		
		reduced_df = single_df['Wor']
		del single_df
		
		#Begin loop through sentences#
		for Sent, Alt in reduced_df.groupby(level=0):
		
			#Begin loop through alternate versions of each sentence#
			for Name, Version in Alt.groupby(level=1):
				
				temp_df = Version.reset_index().drop(['Sent', 'Alt', 'Unit'], axis=1)
				
				for i in range(len(temp_df)):
					temp_index = (int(temp_df.iloc[[i]].values))
					fa.write(str(word_list[temp_index]))
					fa.write(" ")
						
				fa.write("\n")

	fa.close()

	return 
#---------------------------------------------------------------------------------------------#