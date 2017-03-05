#-----------------------------------------------------#
def fold_split(Parameters, input_files):

	import random
	fold_file_dict = {}
	
	training_candidates = int(Parameters.Training_Candidates / Parameters.Lines_Per_File)
	
	if training_candidates < 1: 
		training_candidates = 1
		
	training_search = int(Parameters.Training_Search / Parameters.Lines_Per_File)
	
	if training_search < 1:
		training_search = 1
		
	testing = int(Parameters.Testing / Parameters.Lines_Per_File)
	
	if testing < 1:
		testing = 1
	
	#Distribute files for each fold#
	for fold in range(1,Parameters.CV +1):
	
		fold_file_dict[fold] = {}
	
		#First, randomly select training candidate files#
		if len(input_files) > training_candidates:
			current_training_candidates_files = random.sample(input_files, training_candidates)
			input_files = [x for x in input_files if x not in current_training_candidates_files]
			
		else:
			print("Not enough data to fill desired data distribution.")
			sys.kill()
		
		#Second, randomly select testing files#
		if len(input_files) > testing:
			current_testing_files = random.sample(input_files, testing)
			input_files = [x for x in input_files if x not in current_testing_files]
	
		else:
			print("Not enough data to fill desired data distribution.")
			sys.kill()
			
		#Third, randomly select training search files, looping through#
		current_training_search_files = []
		
		for i in range(1, Parameters.Restarts+1):
			
			if len(input_files) > training_search:
				temp_training_search_files = random.sample(input_files, training_search)
				input_files = [x for x in input_files if x not in temp_training_search_files]
				current_training_search_files.append(temp_training_search_files)
			
			else:
				print("Not enough data to fill desired data distribution.")
				sys.kill()
		
		#Save current fold#
		fold_file_dict[fold]["Training_Candidates"] = current_training_candidates_files
		fold_file_dict[fold]["Training_Search"] = current_training_search_files
		fold_file_dict[fold]["Testing"] = current_testing_files
		
	print("\tInstances remaining after distributing unique sets across folds and restarts: " + str(len(input_files) * Parameters.Lines_Per_File))
	
	return fold_file_dict
#-----------------------------------------------------#