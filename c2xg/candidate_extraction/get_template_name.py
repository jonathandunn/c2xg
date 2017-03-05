#---------------------------------------------------------------------------------------------#
#FUNCTION: get_template_name -----------------------------------------------------------------#
#INPUT: Template with numbers for column labels ----------------------------------------------#
#OUTPUT: List of template units with strings as column labels --------------------------------#
#---------------------------------------------------------------------------------------------#
def get_template_name(template):
	
	#Template names come from annotaton types variable in parameters and are column names#
	
	template_name = []
	
	for i in range(len(template)):
		current_unit = template[i]
		
		template_name.append(current_unit)
	
	return template_name
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#