#---------------------------------------------------------------------------------------------#
#INPUT: List of line dictionaries, input filename (str), encoding_type and docs_per_file------#
#OUTPUT: Write CoNLL file to disk and return list of written filenames -----------------------#
#---------------------------------------------------------------------------------------------#
def write_conll(text_dictionary, 
					input_file, 
					encoding_type
					):

	import csv
	import codecs
	from process_input.get_temp_filename import get_temp_filename

	doc_counter = text_dictionary[0]
	text_dictionary = text_dictionary[1]
	
	base_filename = get_temp_filename(input_file, "")
	actual_filename = base_filename + "." + str(doc_counter) + ".conll"
	
	fw = codecs.open(actual_filename, "w", encoding = encoding_type)
	
	for line_tuple in text_dictionary:
	
		try:
			
			id = line_tuple[0]
			line = line_tuple[1]
		
			for unit in line:
					
				if unit == "<s>":
					fw.write("<s:")
					fw.write(str(id))
					fw.write(">\n")
				
				else:
				
					skip_flag = 0
					
					current_index = unit['index']
					current_word = unit['word']
					current_lemma = current_word
					current_pos = unit['pos']
					
					if current_word != "":
				
						if current_word[0] == "#":
							skip_flag = 1

						if current_word[0] == "@":
							skip_flag = 1
					
						if current_word[0:5] == "EMOJI" and current_word[len(current_word) - 5:] == "EMOJI":
							skip_flag = 1
					
						if "http" in current_word:
							skip_flag = 1
							
						if ":\\" in current_word:
							skip_flag = 1
						
						if skip_flag == 0:
							fw.write(str(current_word) + "\t")
							fw.write(str(current_lemma.lower()) + "\t")
							fw.write(str(current_pos) + "\t")
							fw.write(str(current_index) + "\t")
							fw.write(str("\n"))	
		
		except:
			null_counter = 0
			
	fw.close()
	
	return actual_filename
#---------------------------------------------------------------------------------------------#