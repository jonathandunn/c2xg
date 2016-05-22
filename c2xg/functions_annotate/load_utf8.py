#-------------------------------------------------------------------------------#
#INPUT: Input file and parameters ----------------------------------------------#
#OUTPUT: List of lines ---------------------------------------------------------#
#-------------------------------------------------------------------------------#
def load_utf8(input_file, 
				encoding_type, 
				language, 
				memory_limit, 
				working_directory, 
				use_metadata = False
				):

	import codecs
	
	from functions_annotate.stanford_segment_chinese import stanford_segment_chinese
	from functions_annotate.stanford_segment_arabic import stanford_segment_arabic
	from functions_annotate.strip_metadata import strip_metadata		
	
	line_list = []
	
	#Chinese and Arabic need to be segmented specially using Stanford's segmenter#
	#If the input text is either of these languages, send to processing functions#
	
	if language == "Chinese":
	
		print("Saving Chinese file as UTF-8")
		fo = open(input_file, "rb")
		fw = codecs.open(input_file + ".utf8", "w", encoding = encoding_type, errors = "replace")
		
		for line in fo:
		
			line = line.decode(encoding_type)
			
			if use_metadata == True:
				line = strip_metadata(line)
				
			line = line.replace("\n","").replace("\r","")
			fw.write(str(line))
			fw.write(str(" EOL "))
			
		fo.close()
		fw.close()
		
		print("Segmenting Chinese text.")
		line_list = stanford_segment_chinese(memory_limit, input_file + ".utf8", working_directory, encoding_type)
	
	#Now, if Arabic segementation needed#
	elif language == "Arabic":
	
		print("Saving Arabic file as UTF-8")
		fo = open(input_file, "rb")
		fw = codecs.open(input_file + ".utf8", "w", encoding = encoding_type)
		
		for line in fo:
			line = line.decode(encoding_type)
			
			if use_metadata == True:
				line = strip_metadata(line)
				
			line = line.replace("\n","").replace("\r","")
			fw.write(str(line))
			fw.write(str("\n"))
			
		fo.close()
		fw.close()
		
		print("Segmenting Arabic text.")
		line_list = stanford_segment_arabic(memory_limit, input_file + ".utf8", working_directory, encoding_type)

	#Now if no special segmentation needed#
	else:
			
		fo = open(input_file, "rb")
		counter = 0
		for line in fo:
			counter += 1
			line = line.decode(encoding_type, errors="replace")
			
			if use_metadata == True:
				line = strip_metadata(line)
				
			line_list.append((counter, line))
		fo.close()
			
	return line_list
#--------------------------------------------------#