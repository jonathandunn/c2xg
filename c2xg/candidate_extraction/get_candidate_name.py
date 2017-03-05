#---------------------------------------------------------------------------------------------#
#FUNCTION: get_candidate_name ----------------------------------------------------------------#
#INPUT: Current template and data_file_candidate_constructions -------------------------------#
#OUTPUT: Filename for storing candidates from current template -------------------------------#
#---------------------------------------------------------------------------------------------#
def get_candidate_name(file, data_file_candidate_constructions):
    
	file_name = str(file)
	
	begin_name = file_name.rfind("/")
	file_name = file_name[begin_name+1:]
		
	temp_pickled_name = file_name + ".p"
	pickled_list_file = data_file_candidate_constructions + temp_pickled_name
		
	return pickled_list_file
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#