#-------------------------------------------------------------------------------#
#INPUT: Input file, encoding type ----------------------------------------------#
#OUTPUT: (ID, Meta-Data) tuples ------------------------------------------------#
#-------------------------------------------------------------------------------#
def get_metadata_tuples(input_file, encoding_type):

	import codecs
	
	fo = open(input_file, "rb")
	
	counter = 0
	metadata_list = []
	
	for line in fo:
		
		counter += 1
		line = line.decode(encoding_type, errors="replace")
		line_list = line.split("\t")
		
		#Check to make sure the text is not split into multiple parts, recombine#
		if len(line_list) > 2:
			for i in range(2, len(line_list)):
				line_list[2] += (str(" ") + str(line_list[i]))
				
		current_metadata = line_list[0]
		current_text = line_list[1]
		current_id = counter
	
		#Meta-Data format: Field:Value,Field:Value,Field:Value #
		current_metadata_list = current_metadata.split(",")
		current_meta_dictionary = {}
		
		for field in current_metadata_list:
		
			current_temp = field.split(":")
			current_field = current_temp[0]
			current_value = current_temp[1]
			
			current_meta_dictionary[current_field] = current_value
			
		current_tuple = (current_id, current_meta_dictionary)
		metadata_list.append(current_tuple)			
	
	fo.close()
			
	return metadata_list
#--------------------------------------------------#