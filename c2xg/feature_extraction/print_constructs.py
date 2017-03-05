#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def print_constructs(search_df, 
						candidate, 
						lemma_list, 
						pos_list, 
						category_list, 
						write_examples
						):
	
	import pandas as pd
	import codecs

	fo = codecs.open(write_examples, "a", encoding = "utf-8")
	
	candidate_string = ""
	candidate_flag = 0
	
	#Write candidate name to file#
	for tuple_pair in candidate:
		
		representation = tuple_pair[0]
		index_value = tuple_pair[1]
		
		if representation == "Lex":
			item_value = "'" + str(lemma_list[index_value]) + "'"
		
		elif representation == "Pos":
			item_value = pos_list[index_value].upper()
		
		elif representation == "Cat":
			item_value = "<" + str(category_list[index_value]) + ">"
			
		if candidate_flag > 0:
			candidate_string += " -- "
		
		candidate_string += item_value
		candidate_flag += 1
			
	fo.write(str(candidate_string))
	fo.write(str("\n"))

	#Finished writing candidate name#
	
	#Limit search_df to only lexical representation#
	column_list = search_df.columns
	new_columns = []
	
	for name in column_list:
		if name[0:3] == "Lex":
			new_columns.append(name)
	
	search_df = search_df.loc[:, new_columns]
	#Finished limiting search_df#
	
	for row in search_df.itertuples(index = False, name = "None"):

		for annotation in row:
		
			fo.write(str("\t"))
			fo.write(str(lemma_list[annotation]))
			fo.write(str(" "))
			
		fo.write(str("\n"))
    
	fo.close()
	
	return
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#