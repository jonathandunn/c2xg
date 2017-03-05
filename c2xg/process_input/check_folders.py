#---------------------------------------------------------------------------------------------#
#OUTPUT: None --------------------------------------------------------------------------------#
#For each data file, make sure does not exist before writing current data files --------------#
#---------------------------------------------------------------------------------------------#
def check_folders(input_folder, 
					temp_folder, 
					candidate_folder,
					debug_folder, 
					output_folder,
					dict_folder,
					pos_training_folder,
					pos_testing_folder,
					parameters_folder
					):

	import os.path
	import os
	
	if os.path.isdir(input_folder) == False:
		os.makedirs(input_folder)
		print("Creating input folder")
			
	if os.path.isdir(output_folder) == False:
		os.makedirs(output_folder)
		print("Creating output folder")
		
	if os.path.isdir(parameters_folder) == False:
		os.makedirs(parameters_folder)
		print("Creating parameters folder")
		
	if os.path.isdir(temp_folder) == False:
		os.makedirs(temp_folder)
		print("Creating temp folder")
		
	if os.path.isdir(candidate_folder) == False:
		os.makedirs(candidate_folder)
		print("Creating candidate folder")
		
	if os.path.isdir(debug_folder) == False:
		os.makedirs(debug_folder)
		print("Creating debug folder")
		
	if os.path.isdir(dict_folder) == False:
		os.makedirs(dict_folder)
		print("Creating dict folder")
		
	if os.path.isdir(pos_training_folder) == False:
		os.makedirs(pos_training_folder)
		print("Creating POS training folder")
		
	if os.path.isdir(pos_testing_folder) == False:
		os.makedirs(pos_testing_folder)
		print("Creating POS testing folder")
			
	return
#---------------------------------------------------------------------------------------------#
