#--------------------------------------------#
#--- Convert CONLL files to training_dfs ----#
#--------------------------------------------#
def create_training_files(file_list, model, temp_dir, output_name):

	import codecs
	import os
	import sys
	
	#Change path for importing c2xg modules#
	os.chdir("../")
	sys.path.append(os.path.abspath(""))

	from c2xg.functions_input.pandas_open import pandas_open
	from c2xg.functions_constituent_reduction.expand_sentences import expand_sentences
	from c2xg.functions_candidate_extraction.read_candidates import read_candidates
	from c2xg.functions_candidate_extraction.write_candidates import write_candidates
	
	os.chdir("./c2xg/")
	sys.path.append(os.path.abspath(""))
	os.chdir("./../utilities")

	#First, load constituent grammar model#
	current_dictionary = read_candidates(model)
		
	pos_list = current_dictionary['pos_list']
	lemma_list = current_dictionary['lemma_list']
	category_list = current_dictionary['category_list']
	word_list = current_dictionary['word_list']
	phrase_constituent_list = current_dictionary['phrase_constituent_list']
	semantic_category_dictionary = current_dictionary['semantic_category_dictionary']
	lemma_dictionary = current_dictionary['lemma_dictionary']
	pos_dictionary = current_dictionary['pos_dictionary']
	category_dictionary = current_dictionary['category_dictionary']
	emoji_dictionary = current_dictionary['emoji_dictionary']
	
	#Second, open individual CONLL files, write to single output file#
	fw = codecs.open(temp_dir + output_name, "w", encoding = "utf-8")
	counter = 0
	
	#Loop through individual CONLL files#
	for file in file_list:
		
		fo = codecs.open(temp_dir + file, "r", encoding = "utf-8")
		
		for line in fo:
			
			if line[0:2] == "<s":
				counter += 1
				fw.write("<s:" + str(counter) + ">\n")
				
			else:
				fw.write(line)
			
		fo.close()	
	fw.close()
	
	#Third, load merged CONLL file as dataframe#
	encoding_type = "utf-8"
	
	input_df = pandas_open(temp_dir + output_name, 
							encoding_type, 
							semantic_category_dictionary, 
							word_list, 
							lemma_list, 
							pos_list, 
							lemma_dictionary, 
							pos_dictionary, 
							category_dictionary,
							write_output = False,
							delete_temp = False
							)
										
	total_words = len(input_df)
	
	input_df = expand_sentences(input_df, 
									lemma_list, 
									pos_list, 
									category_list, 
									encoding_type, 
									write_output = False, 
									phrase_constituent_list = phrase_constituent_list
									)
	
	#Now pickle expanded DataFrame to file#
	write_name = temp_dir + output_name + ".df"
	write_candidates(write_name, [input_df, total_words])

	return
#----------------------------------------------------------------------------------------------------#

output_name = "Test.conll"

temp_dir = "../../../../data/Input/Temp/"
model = "../../../../data/Output/English.Caveice.1.Constituents.model"

file_list = [
"English.Caveice.1095.txt.1.conll",
"English.Caveice.1096.txt.1.conll",
]
	
create_training_files(file_list, model, temp_dir, output_name)