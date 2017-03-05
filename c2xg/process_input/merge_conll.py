#--------------------------------------------#
def merge_conll(file_tuple, encoding_type):

	import codecs

	counter = 0
	
	file_list = file_tuple[0]
	output_name = file_tuple[1]
	
	fw = codecs.open(output_name, "w", encoding = encoding_type)
	
	for file in file_list:
	
		fo = codecs.open(file, "r", encoding = encoding_type)
		
		for line in fo:
		
			if line[0] == "<":
				counter += 1
				fw.write("<s:" + str(counter) + ">\n")
				
			else:
				fw.write(line)
		
		fo.close()
		
	fw.close()

	return
#--------------------------------------------#