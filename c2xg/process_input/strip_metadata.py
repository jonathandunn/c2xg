#-------------------------------------------------------------------------------#
#INPUT: Input file, encoding type ----------------------------------------------#
#OUTPUT: (ID, Meta-Data) tuples ------------------------------------------------#
#-------------------------------------------------------------------------------#
def strip_metadata(line):

	line_list = line.split("\t")
		
	#Check to make sure the text is not split into multiple parts, recombine#
	if len(line_list) > 2:
		for i in range(2, len(line_list)):
			line_list[2] += (str(" ") + str(line_list[i]))
				
	current_text = line_list[1]

	return current_text
#--------------------------------------------------#