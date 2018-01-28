#----------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------#
def pandas2arff(df, filename, wekaname = "pandasdata", cleanstringdata = True, cleannan = True):
	
	
	"""
	AUTHOR: GitHub/saurabhnagrecha/Pandas-to-ARFF
	converts the pandas dataframe to a weka compatible file
	df: dataframe in pandas format
	filename: the filename you want the weka compatible file to be in
	wekaname: the name you want to give to the weka dataset (this will be visible to you when you open it in Weka)
	cleanstringdata: clean up data which may have spaces and replace with "_", special characters etc which seem to annoy Weka. 
					 To suppress this, set this to False
	cleannan: replaces all nan values with "?" which is Weka's standard for missing values. 
			  To suppress this, set this to False
	"""
	
	import re
	import numpy as np
	
	def cleanstring(s):
		if s!="?":
			return re.sub('[^A-Za-z0-9]+', "_", str(s))
		else:
			return "?"
			
	dfcopy = df #all cleaning operations get done on this copy

	
	if cleannan!=False:
		dfcopy = dfcopy.fillna(-999999999) #this is so that we can swap this out for "?"
		#this makes sure that certain numerical columns with missing values don't get stuck with "object" type
 
	f = open(filename,"w")
	arffList = []
	arffList.append("@relation " + wekaname + "\n")
	#look at each column's dtype. If it's an "object", make it "nominal" under Weka for now (can be changed in source for dates.. etc)
	for i in range(df.shape[1]):
		if dfcopy.dtypes[i]=='O' or (df.columns[i] in ["Class","CLASS","class"]):
			if cleannan!=False:
				dfcopy.iloc[:,i] = dfcopy.iloc[:,i].replace(to_replace=-999999999, value="?")
			if cleanstringdata!=False:
				dfcopy.iloc[:,i] = dfcopy.iloc[:,i].apply(cleanstring)
			_uniqueNominalVals = [str(_i) for _i in np.unique(dfcopy.iloc[:,i])]
			_uniqueNominalVals = ",".join(_uniqueNominalVals)
			_uniqueNominalVals = _uniqueNominalVals.replace("[","")
			_uniqueNominalVals = _uniqueNominalVals.replace("]","")
			_uniqueValuesString = "{" + _uniqueNominalVals +"}" 
			arffList.append("@attribute " + str(df.columns[i]) + _uniqueValuesString + "\n")
		else:
			arffList.append("@attribute " + str(df.columns[i]) + " real\n") 
			#even if it is an integer, let's just deal with it as a real number for now
	arffList.append("@data\n")		   
	for i in range(dfcopy.shape[0]):#instances
		_instanceString = ""
		for j in range(df.shape[1]):#features
				if dfcopy.dtypes[j]=='O':
					_instanceString+="\"" + str(dfcopy.iloc[i,j]) + "\""
				else:
					_instanceString+=str(dfcopy.iloc[i,j])
				if j!=dfcopy.shape[1]-1:#if it's not the last feature, add a comma
					_instanceString+=","
		_instanceString+="\n"
		if cleannan!=False:
			_instanceString = _instanceString.replace("-999999999.0","?") #for numeric missing values
			_instanceString = _instanceString.replace("\"?\"","?") #for categorical missing values
		arffList.append(_instanceString)
	f.writelines(arffList)
	f.close()
	del dfcopy
	return True

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
	
	vector_list = []
	
	#Loop through files#
	for file in file_list:
		
		print("Opening " + str(file))
		
		print("\tGetting column names.")
		current_columns = read_candidates(file + ".Columns")
		
		print("\tLoading vectors.")
		current_vector = pd.read_hdf(file, key="Table")
		
		vector_list.append(current_vector)
	#Done looping through files#
	
	print("Now joining vectors into single DataFrame.")
	vector_df = pd.concat(vector_list, axis = 0, ignore_index = True)
	
	print("Now saving to ARFF file.")
	pandas2arff(vector_df, "Output.arff", wekaname = "pandasdata", cleanstringdata = True, cleannan = True)

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