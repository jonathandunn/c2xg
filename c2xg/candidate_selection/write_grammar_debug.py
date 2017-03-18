#---------------------------------------------------------------------------------#
def write_grammar_debug(final_grammar, suffix, Grammar, Parameters):

	#Write readable grammar to file#
	import codecs
	
	debug_file = Parameters.Debug_File + suffix
	fw = codecs.open(debug_file, "w", encoding = Parameters.Encoding_Type)
	
	for construction in final_grammar:

		for unit in construction:

			type = unit[0]
			value = unit[1]
			
			if type == "Cat":
				value = "<" + str(Grammar.Category_List[value]) + ">"
			
			elif type == "Pos":
				value = str(Grammar.POS_List[value]).upper()
				
			elif type == "Lex":
				value = "'" + str(Grammar.Lemma_List[value]) + "'"
				
			fw.write(str(value) + " ")
			
		fw.write("\n")
	fw.close()
	
	return
#---------------------------------------------------------------------------------#