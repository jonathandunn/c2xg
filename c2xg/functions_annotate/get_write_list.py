#-------------------------------------------------------------------------#
def get_write_list(text_dictionary, docs_per_file):

	number_of_files = int(len(text_dictionary) / docs_per_file) + 1
		
	write_list = []
	start_index = 0
	end_index = docs_per_file 
		
	if end_index > len(text_dictionary):
		end_index = len(text_dictionary) 
			
	for i in range(1,number_of_files+1):
		temp_tuple = (i, start_index, end_index)
		write_list.append(temp_tuple)
		start_index += docs_per_file
		end_index += docs_per_file
		
		if end_index > len(text_dictionary):
			end_index = len(text_dictionary) 
			
	temp_list = []
	
	for segment in write_list:
	
		doc_id = segment[0]
		start_index = segment[1]
		end_index = segment[2]
		
		temp_dictionary = text_dictionary[start_index:end_index]
		temp_tuple = (doc_id, temp_dictionary)
		temp_list.append(temp_tuple)	
			
	return temp_list
#------------------------------------------------------------------------#