#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
def get_sent_examples(data_file,
						output_file,
						rule_dict,
						constituent_len_dictionary,
						length_list,
						pos_list,
						encoding_type,
						semantic_category_dictionary,
						word_list,
						lemma_list,
						lemma_dictionary,
						pos_dictionary,
						category_dictionary,
						delete_temp
						):

	import pandas as pd
	import cytoolz as ct
	import time
		
	from functions_input.pandas_open import pandas_open
	from functions_constituent_reduction.find_unit_index import find_unit_index
	from functions_constituent_reduction.get_search_df_expansion import get_search_df_expansion
	from functions_constituent_reduction.get_expansion_query import get_expansion_query
	from functions_constituent_reduction.constituents_reduce import constituents_reduce
		
	time_start = time.time()
	
	print(data_file)
	data_file = str("../../../../data/Input/Temp/ukWac (51).conll")
	print(data_file)
	
	#Loop through files to support out-of-memory datasets#
	current_df = pandas_open(data_file, 
								encoding_type,
								semantic_category_dictionary,
								word_list,
								lemma_list,
								pos_list,
								lemma_dictionary,
								pos_dictionary,
								category_dictionary,
								save_words = True,
								write_output = False,
								delete_temp = delete_temp
								)
	
	import codecs
	
	fw = codecs.open(output_file, "w", encoding = "utf-8")
	
	bracket_dictionary = {}
	
	print("Initializing dict.")
	for mas in list(current_df.loc[:,"Mas"].values):
		bracket_dictionary[mas] = {}
		bracket_dictionary[mas]["start"] = ""
		bracket_dictionary[mas]["end"] = ""
	
	print("\tFinding constituent matches.")	
	#Loop through constituents by length, creating only 1 search_df for each length#
	for length in length_list:
	
		print("Starting length: " + str(length) + " with " + str(len(constituent_len_dictionary[length])) + " sequences.")
		
		#Generate initial search DF#
		copy_df = current_df.copy("Deep")
		search_df = get_search_df_expansion(copy_df, length)
		
		#Loop through constituents of current length#
		for constituent in constituent_len_dictionary[length]:
		
			direction = rule_dict[constituent]

			if direction == "L":
				current_head = constituent[0]
			
			elif direction == "R":
				current_head = constituent[-1]
			
			try:
				head_write = pos_list[current_head]
				
			except:
				head_write = "na"
				
			constituent_write = ""
			
			for thing in constituent:
				try:
					constituent_write += str(pos_list[thing]) + " "
				except:
					constituent_write += "na "

			#Find constituents#
			query_string = get_expansion_query(constituent)
			match_df = search_df.query(query_string, parser='pandas', engine='numexpr')
			
			mas_values = list(match_df.loc[:,"Mas"].values)

			for start_mas in mas_values:
			
				end_mas = start_mas + length - 1
				
				bracket_dictionary[start_mas]["start"] += "["
				bracket_dictionary[end_mas]["end"] += "]"
	
	previous_sent = 0	
				
	for row in current_df.itertuples(index = False, name = None):
		
		current_sent = row[0]
		current_mas = row[6]
		current_word = row[2]
		
		if current_sent != previous_sent:
			previous_sent = current_sent
			fw.write("\n")
			
		fw.write(str(bracket_dictionary[current_mas]["start"]))
		fw.write(str(current_word))
		fw.write(str(bracket_dictionary[current_mas]["end"]))
		fw.write(str(" "))
			
			
		#Done looping through constituents of current length#
	#Done looping through constituents by length#			
	fw.close()
	print("Time for finding examples by sentence: " + str(time.time() - time_start))
	
	return 
#---------------------------------------------------------------------------------------------#