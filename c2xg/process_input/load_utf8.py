#-------------------------------------------------------------------------------#
#INPUT: Input file and parameters ----------------------------------------------#
#OUTPUT: List of lines ---------------------------------------------------------#
#-------------------------------------------------------------------------------#
def load_utf8(input_file, Parameters):

	import codecs
	from process_input.strip_metadata import strip_metadata		
	
	line_list = []
	
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