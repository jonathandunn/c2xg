#---------------------------------------------------------------------------------------------#
#FUNCTION: get_column_list-----------------------------------------------------------------------#
#INPUT: template -----------------------------------------------------------------------------#
#OUTPUT: Take template, return list of columns to include in candidate search DataFrame ---------#
#---------------------------------------------------------------------------------------------#
def get_column_list(template):
	
	column_list = [0, template[0]]
	column_names = ["c1", "c2"]
	column_counter = 2
	
	for i in range(len(template) - 1):
		column_list.append(template[i + 1])
		
		column_names.append("c" + str(column_counter + 1))
		
		column_counter += 1

	return [column_list, column_names]
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#