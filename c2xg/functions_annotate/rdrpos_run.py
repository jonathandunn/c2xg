#-------------------------------------------------------------------------------#
#---run_rdrpos -----------------------------------------------------------------#
#INPUT: Line tuple (ID, Str), encoding, rdr object, dict file, language name ---#
#OUTPUT: Annotated data for writing to CoNLL file ------------------------------#
#-------------------------------------------------------------------------------#
def rdrpos_run(line_tuple, 
				encoding_type, 
				r, 
				DICT, 
				language
				):
		
	counter = 0
	
	id = line_tuple[0]
	line = line_tuple[1]
	
	try:
		if len(line) > 1:
		
			if language == "English":
				line_annotated = r.tagRawEnSentence(DICT, line)
				line_annotated = line_annotated.split()

			elif language == "Vietnamese":
				line_annotated = r.tagRawVnSentence(DICT, line)
				line_annotated = line_annotated.split()
				
			else:
				line_annotated = r.tagRawSentence(DICT, line)
				line_annotated = line_annotated.split()
				
			#Now prepare line for adding to line list#
			current_line = ["<s>"]
			
			for unit in line_annotated:
				
				counter += 1
				
				divider_index = unit.rfind("/")
				temp_word = unit[:divider_index]
				temp_pos = unit[divider_index+1:].lower()
				
				current_unit = {}
				
				if len(temp_word) > 0:
					current_unit['word'] = temp_word
					current_unit['pos'] = temp_pos
					current_unit['index'] = counter
					current_line.append(current_unit)	

			return (id, current_line)
		
		else:
			return
			
	except:
		return
#--------------------------------------------------#





