#-------------------------------------------------------------------------------#
#INPUT: Line tuple (ID, String), pos-tagging model name ------------------------#
#OUTPUT: Annotated line tuple (ID, Annotation) ---------------------------------#
#-------------------------------------------------------------------------------#
def stanford_run(line_tuple, pos_model):

	import requests
	import codecs
	
	id = line_tuple[0]
	line = line_tuple[1]
	
	current_line = ["<s>"]
	process_marker = 0
	fail_count = 0
	
	if line:
	
		while process_marker == 0:
		
			url_command = 'http://localhost:9000/?properties={"annotators": "tokenize, ssplit, pos", '
			url_command += '"tokenize.whitespace": "true", '
			url_command += '"ssplit.eolonly": "true", '
			url_command += '"pos.model": "' + pos_model + '", '
			url_command += '"outputFormat": "text", "encoding": "utf-8"}'
		
			r = requests.post(url_command, 
								line.encode("utf-8"), 
								headers = {"Content-Type": "charset=UTF-8"}
								)
		
			line_annotated = r.content
			line_annotated = line_annotated.decode("utf-8")
			line_annotated = line_annotated.split("\n")
			
			if "chinese" in pos_model or "arabic" in pos_model:
				line_annotated = line_annotated[:len(line_annotated)-2]
			
			try:
				index_counter = 1
				for unit in line_annotated:
					
					unit = unit.replace("\r","").replace("\n","").replace("[","").replace("]","").replace("\0","")
					
					unit_dictionary = {}
					if unit[0:4] == "Text":

						temp_list = unit.split(" ")
						
						current_word = temp_list[0].replace("Text=","")
						unit_dictionary['word'] = current_word
						
						if current_word != "":
							unit_dictionary['pos'] = temp_list[3].replace("PartOfSpeech=","").lower()
							unit_dictionary['index'] = index_counter
						
							index_counter += 1
							current_line.append(unit_dictionary)
				
				#Done parsing Stanford output, now end loop#
				process_marker = 1
			
			except:
				
				fail_count += 1
				
				if fail_count > 3:
					print("\tFail count exceeded, skipping line.")
					current_line = []
					process_marker = 1
			
		return (id, current_line)
		
	else:
	
		return
#--------------------------------------------------#