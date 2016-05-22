#-------------------------------------------------------------------------------#
def write_debug(category_frequency, lemma_frequency, word_frequency, pos_frequency, debug_file, encoding_type):

	fdebug = open(debug_file + "Category", "w", encoding=encoding_type)
	
	for key in category_frequency.keys():
		fdebug.write(str(key))
		fdebug.write(": ")
		fdebug.write(str(category_frequency[key]))
		fdebug.write("\n")
	
	fdebug.close()
	
	fdebug = open(debug_file + "Lemma", "w", encoding=encoding_type)
	
	for key in lemma_frequency.keys():
		fdebug.write(str(key))
		fdebug.write(": ")
		fdebug.write(str(lemma_frequency[key]))
		fdebug.write("\n")
	
	fdebug.close()
		
	fdebug = open(debug_file + "Word", "w", encoding=encoding_type)
	
	for key in word_frequency.keys():
		fdebug.write(str(key))
		fdebug.write(": ")
		fdebug.write(str(word_frequency[key]))
		fdebug.write("\n")
	
	fdebug.close()
	
	fdebug = open(debug_file + "POS", "w", encoding=encoding_type)
	
	for key in pos_frequency.keys():
		fdebug.write(str(key))
		fdebug.write(": ")
		fdebug.write(str(pos_frequency[key]))
		fdebug.write("\n")
	
	fdebug.close()
	
	return
#------------------------------------------------------------------------------#