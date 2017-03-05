#---------------------------------------------------------------------------------------------#
#INPUT: list of candidates (frequency reduced), list of lemma and pos and category indexes ---#
#--------and file name for debug info --------------------------------------------------------#
#OUTPUT: Write a file with human readable construction candidates for debugging --------------#
#---------------------------------------------------------------------------------------------#	
def candidate_debug(candidate_list, 
						lemma_list, 
						pos_list, 
						category_list, 
						candidate_debug_file, 
						encoding_type
						):
	
	fw = open(candidate_debug_file, "w", encoding=encoding_type)
		
	for candidate in candidate_list:
		
		for pair in candidate:
			column = pair[0]
			index = pair[1]
				
			if column == "Lex":
				value = lemma_list[index]
					
			elif column == 'Pos':
				value = pos_list[index]
					
			elif column == 'Cat':
				value = category_list[index]
					
			fw.write(str(value))
			fw.write(" ")
		
		#End current construction candidate, add newline#
		fw.write("\n")
	
	return
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#