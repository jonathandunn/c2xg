#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def read_candidates(file):
    
	import pickle
	
	candidate_list = []
	
	with open(file,'rb') as f:
		candidate_list = pickle.load(f)
		
	return candidate_list
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def to_csv(file_list):
    
	import pandas as pd
	
	for file in file_list:
		
		print("Opening " + str(file))
		
		print("\tGetting column names.")
		current_columns = read_candidates(file + ".Columns")
		
		print("\tLoading vectors.")
		current_vector = pd.read_hdf(file, key="Table")
		current_vector.columns = current_columns
		
		print("\tWriting zipped CSV.")
		current_vector.to_csv(file + ".csv", compression = "gzip")

	return
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
file_list = [
"English.Dialect (1).txt.1.conll.Features",
"English.Dialect (2).txt.1.conll.Features",
"English.Dialect (3).txt.1.conll.Features",
"English.Dialect (4).txt.1.conll.Features",
"English.Dialect (5).txt.1.conll.Features",
"English.Dialect (6).txt.1.conll.Features",
"English.Dialect (7).txt.1.conll.Features",
"English.Dialect (8).txt.1.conll.Features",
"English.Dialect (9).txt.1.conll.Features",
"English.Dialect (10).txt.1.conll.Features"
]

to_csv(file_list)