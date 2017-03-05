#---------------------------------------------------------------------------------------------#
#INPUT: List of line dictionaries, input filename (str), encoding_type and docs_per_file------#
#OUTPUT: Write CoNLL file to disk and return list of written filenames -----------------------#
#---------------------------------------------------------------------------------------------#
def write_conll_raw(input_file, Parameters):

	import codecs
	
	from process_input.replace_emojis import replace_emojis
	from process_input.tokenize_line import tokenize_line
	from process_input.get_temp_filename import get_temp_filename
	from process_input.check_data_files import check_data_files
	
	print("\tConverting raw text to untagged CoNLL format for " + str(input_file))
	output_file = get_temp_filename(input_file, ".conll_raw")
	
	#Define punctuation#
	punc_list = [
	'"', ']', '[', ',', ')', '(', '>', '#', '_', '•',
	'<', '&', '. ', "' ", '/ ', '“', "'", '-', '...',
	'’ ', '^', '\0', '*', ': ', '@ ', '­', '..',
	"# ", '+', '=', '~', '?', '!', '` ', '%', ':',
	'”', '\n', '` ', '…', '·', ';', '.', '\\'
	]
	
	counter = 0
	doc_counter = 1
	
	output_files = []
	
	fo = codecs.open(input_file, "rb")
	fw = codecs.open(output_file + "." + str(doc_counter), "w", encoding = Parameters.Encoding_Type)
	
	for line in fo:
	
		line = line.decode(encoding = Parameters.Encoding_Type, errors = "replace")
		line = line.replace("\n","").replace("\r","")
		
		#First, check if new file needs to be opened#
		if counter >= Parameters.Lines_Per_File:
			
			output_files += [output_file + "." + str(doc_counter)]
			doc_counter += 1
			counter = 0
			fw.close()
			fw = codecs.open(output_file + "." + str(doc_counter), "w", encoding = Parameters.Encoding_Type) 
		#Done checking file size#
		
		if line != None:
			
			line = replace_emojis(line, Parameters)
			line = tokenize_line(line)
			
			if line != None:
			
				line_list = line.split(" ")
				
				if len(line_list) > 1:
			
					counter += 1
					id_string = "<s:" + str(counter) + ">"
					fw.write(id_string + "\t" + id_string + "\t" + id_string + "\t" + id_string + "\t" + id_string + "\n")
						
					for unit in line_list:
					
						if unit != "" and unit != None:

							if unit[0:5] == "EMOJI" and unit[len(unit) - 5:] == "EMOJI":
								current_word = unit[5:len(unit)-5]
								current_word = "{" + current_word + "}"
							
							elif "http" in unit:
								current_word = ""
								
							elif unit in punc_list:
								current_word = ""
							
							else:
								current_word = unit
					
						if current_word != "" and current_word != None:
							
							fw.write(str(current_word.replace("\0", "NULL")))
							fw.write("\t")
							
							fw.write(str(current_word).replace("\0", "NULL").lower())
							fw.write("\t")
							
							fw.write("A\t")
							fw.write("B\t")
							fw.write("C\n")

	fo.close()
	fw.close()
	
	#Delete last, unfinished file#
	check_data_files(output_file + "." + str(doc_counter))
	
	return output_files
#---------------------------------------------------------------------------------------------#