#--------------------------------------------------------------------------------------#
def split_list(seq, num):
  
	avg = len(seq) / float(num)
	out = []
	last = 0.0

	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg

	return out
#--------------------------------------------------------------------------------------#

#--------------------------------------------------------------------------------------#
def merge_conll_names(training_testing_files, testing_files, Parameters):
	
	tuple_list = []
	training_list = []
	testing_list = []
	counter = 0

	for file_list in training_testing_files:
		
		counter += 1
		current_filename = Parameters.Temp_Folder + "/" + Parameters.Nickname + ".Restart." + str(counter) + ".conll"
		tuple_list.append((file_list, current_filename))
		training_list.append(current_filename)
				
	current_filename = Parameters.Temp_Folder + "/" + Parameters.Nickname + ".Testing.conll"
	tuple_list.append((testing_files, current_filename))
	testing_list.append(current_filename)

	return tuple_list, training_list, testing_list