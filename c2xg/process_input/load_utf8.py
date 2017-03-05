#-------------------------------------------------------------------------------#
#INPUT: Input file and parameters ----------------------------------------------#
#OUTPUT: List of lines ---------------------------------------------------------#
#-------------------------------------------------------------------------------#
def load_utf8(input_file, Parameters):

	import codecs
	from process_input.stanford_segment_chinese import stanford_segment_chinese
	from process_input.stanford_segment_arabic import stanford_segment_arabic
	from process_input.strip_metadata import strip_metadata		
	
	line_list = []
	
	#Chinese and Arabic need to be segmented specially using Stanford's segmenter#
	#If the input text is either of these languages, send to processing functions#
	
	if Parameters.Language == "Chinese":
	
		print("Saving Chinese file as UTF-8")
		fo = open(input_file, "rb")
		fw = codecs.open(input_file + ".utf8", "w", encoding = Parameters.Encoding_Type, errors = "replace")
		
		for line in fo:
		
			line = line.decode(Parameters.Encoding_Type)
			
			if Parameters.Use_Metadata == True:
				line = strip_metadata(line)
				
			line = line.replace("\n","").replace("\r","")
			fw.write(str(line))
			fw.write(str(" EOL "))
			
		fo.close()
		fw.close()
		
		print("Segmenting Chinese text.")
		line_list = stanford_segment_chinese(Parameters, input_file + ".utf8")
	
	#Now, if Arabic segementation needed#
	elif Parameters.Language == "Arabic":
	
		print("Saving Arabic file as UTF-8")
		fo = open(input_file, "rb")
		fw = codecs.open(input_file + ".utf8", "w", encoding = Parameters.Encoding_Type)
		
		for line in fo:
			line = line.decode(Parameters.Encoding_Type)
			
			if Parameters.Use_Metadata == True:
				line = strip_metadata(line)
				
			line = line.replace("\n","").replace("\r","")
			fw.write(str(line))
			fw.write(str("\n"))
			
		fo.close()
		fw.close()
		
		print("Segmenting Arabic text.")
		line_list = stanford_segment_arabic(Parameters, input_file + ".utf8")

	#Now if no special segmentation needed#
	else:
			
		fo = open(input_file, "rb")
		counter = 0
		for line in fo:
			counter += 1
			line = line.decode(Parameters.Encoding_Type, errors="replace")
			
			if Parameters.Use_Metadata == True:
				line = strip_metadata(line)
				
			line_list.append((counter, line))
		fo.close()
			
	return line_list
#--------------------------------------------------#