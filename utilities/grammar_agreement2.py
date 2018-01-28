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
def write_candidates(file, candidate_list):
    
	import pickle
	import os.path
	import os
	
	if os.path.isfile(file):
		os.remove(file)
	
	with open(file,'wb') as f:
		pickle.dump(candidate_list,f)
	
	return
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def grammar_agreement(grammar1_file, grammar2_file):

	grammar1_dict = read_candidates(grammar1_file)
	grammar2_dict = read_candidates(grammar2_file)
	
	grammar1 = grammar1_dict["candidate_list"]
	grammar2 = grammar2_dict["candidate_list"]
	
	shared = 0
	
	for unit in grammar1:
		if unit in grammar2:
			shared += 1
	
	print(grammar1_file + " and " + grammar2_file)
	print("Total shared: " + str(shared))
	print("Total in grammar 1: " + str(len(grammar1)))
	print("Agreement: " + str(float(shared) / len(grammar1)))
	print("")
	print("")

	
	return 
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
grammar_list = [
"EastAfrica.2.Constructions.model",
"HongKong.2.Constructions.model",
"India.2.Constructions.model",
"Ireland.2.Constructions.model"
]

for i in [0, 1, 2, 3]:
	for j in [0, 1, 2, 3]:
	
		if i != j:
			grammar1_file = grammar_list[i]
			grammar2_file = grammar_list[j]
			grammar_agreement(grammar1_file, grammar2_file)