#--------------------------------------------#
#-- Generate dictionary of possible settings -#
#--------------------------------------------#
def get_parameters(name_list):

	parameter_list = []
	
	for name in name_list:

		parameter_dictionary = {}
		name_list = name.split(",")
		
		for pair in name_list:
			
			try:
				pair = pair.split(":")
				feature = pair[0]
				value = pair[1]
				parameter_dictionary[feature] = value
			
			except:
				null_counter = 0
		
		if "pairwise_lr" not in parameter_dictionary:
			parameter_dictionary["pairwise_lr"] = "Off"
		
		if "pairwise_rl" not in parameter_dictionary:
			parameter_dictionary["pairwise_rl"] = "Off"
			
		if "summed_lr" not in parameter_dictionary:
			parameter_dictionary["summed_lr"] = "Off"
			
		if "summed_rl" not in parameter_dictionary:
			parameter_dictionary["summed_rl"] = "Off"
			
		if "mean_lr" not in parameter_dictionary:
			parameter_dictionary["mean_lr"] = "Off"
			
		if "mean_rl" not in parameter_dictionary:
			parameter_dictionary["mean_rl"] = "Off"
			
		if "reduced_lr" not in parameter_dictionary:
			parameter_dictionary["reduced_lr"] = "Off"
			
		if "reduced_rl" not in parameter_dictionary:
			parameter_dictionary["reduced_rl"] = "Off"
			
		if "divided_lr" not in parameter_dictionary:
			parameter_dictionary["divided_lr"] = "Off"
			
		if "divided_rl" not in parameter_dictionary:
			parameter_dictionary["divided_rl"] = "Off"
			
		if "changed_scalar" not in parameter_dictionary:
			parameter_dictionary["changed_scalar"] = "Off"
			
		if "changed_categorical" not in parameter_dictionary:
			parameter_dictionary["changed_categorical"] = "Off"
			
		if "freq" not in parameter_dictionary:
			parameter_dictionary["freq"] = "Off"

		parameter_list.append(parameter_dictionary)
														
	return parameter_list
#-----------------------------------------------#