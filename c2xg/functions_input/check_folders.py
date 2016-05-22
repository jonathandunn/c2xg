#---------------------------------------------------------------------------------------------#
#OUTPUT: None --------------------------------------------------------------------------------#
#For each data file, make sure does not exist before writing current data files --------------#
#---------------------------------------------------------------------------------------------#
def check_folders(input_folder, 
					temp_folder, 
					debug_folder, 
					output_folder
					):

	import os.path
	import os
	
	print("Checking folders.")
	
	if os.path.isdir(input_folder) == False:
		os.makedirs(input_folder)
		print("Creating input folder")
			
	if os.path.isdir(output_folder) == False:
		os.makedirs(output_folder)
		print("Creating output folder")
		
	if os.path.isdir(temp_folder) == False:
		os.makedirs(temp_folder)
		print("Creating temp folder")
		
	if os.path.isdir(debug_folder) == False:
		os.makedirs(debug_folder)
		print("Creating debug folder")
			
	return
#---------------------------------------------------------------------------------------------#
