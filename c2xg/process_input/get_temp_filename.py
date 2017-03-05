def get_temp_filename(input_file, 
						suffix,
						candidate_flag = False
						):

	slash_index = input_file.rfind("/")
	
	base_input_file = input_file[slash_index + 1:]
	input_file_path = input_file[:slash_index]
	
	if candidate_flag == False:
		if "/Temp/" in input_file:
			output_file = input_file + suffix
		
		else:
			output_file = input_file_path + "/Temp/" + base_input_file + suffix
			
	elif candidate_flag == True:
		if "/Candidates/" in input_file:
			output_file = input_file + suffix
		
		else:
			output_file = input_file_path + "/Candidates/" + base_input_file + suffix
		
	return output_file