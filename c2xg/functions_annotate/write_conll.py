#---------------------------------------------------------------------------------------------#
#INPUT: List of line dictionaries, input filename (str), encoding_type and docs_per_file------#
#OUTPUT: Write CoNLL file to disk and return list of written filenames -----------------------#
#---------------------------------------------------------------------------------------------#
def write_conll(text_dictionary, 
					input_file, 
					encoding_type, 
					docs_per_file
					):

	import csv
	import codecs
	from functions_input.get_temp_filename import get_temp_filename
	
	doc_counter = 1
	line_counter = 0
	
	base_filename = get_temp_filename(input_file, "")
	
	fw = codecs.open(base_filename + ".1.conll", "w", encoding = encoding_type)
	
	for line_tuple in text_dictionary:
	
		try:
			line_counter += 1
			
			id = line_tuple[0]
			line = line_tuple[1]
		
			if line_counter == docs_per_file:
				doc_counter += 1
				line_counter = 0
				fw.close()
				write_string = base_filename + "." + str(doc_counter) + ".conll"
				fw = codecs.open(write_string, "a", encoding = encoding_type)
			
			for unit in line:
					
				if unit == "<s>":
					fw.write("<s:")
					fw.write(str(id))
					fw.write(">\n")
				
				else:
				
					current_index = unit['index']
					current_word = unit['word']
					current_lemma = current_word
					current_pos = unit['pos']
					
					if current_word != "":
				
						if len(current_pos) > 3:
							current_pos = current_pos[0:3]
					
						if current_word[0] == "#":
							current_pos = "ht"
							current_lemma = current_lemma[1:]

						if current_word[0] == "@":
							current_pos = "at"
							current_lemma = current_lemma[1:]
					
						if current_word[0:5] == "EMOJI" and current_word[len(current_word) - 5:] == "EMOJI":
							current_pos = "em"
							current_word = current_word[5:len(current_word)-5]
							current_word = "{" + current_word + "}"
							current_lemma = current_word.lower()
					
						if "http" in current_word:
							current_pos = "url"
							current_lemma = "url"
						
						fw.write(str(current_word) + "\t")
						fw.write(str(current_lemma.lower()) + "\t")
						fw.write(str(current_pos) + "\t")
						fw.write(str(current_index) + "\t")
						fw.write(str("\n"))	
		
		except:
			null_counter = 0
			
	fw.close()
	
	#Create list of written CoNLL formatted files#
	conll_files = []
	for i in range(doc_counter):
		write_string = base_filename + "." + str(i+1) + ".conll"
		conll_files.append(write_string)
		
	return conll_files
#---------------------------------------------------------------------------------------------#