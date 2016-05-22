#-----C2xG, v 1.0 ----------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#---- Copyright, 2015-2016 Jonathan E. Dunn --------------------------------------------------#
#---------- www.jdunn.name -------------------------------------------------------------------#
#---------- jonathan.edwin.dunn@gmail.com ----------------------------------------------------#
#---------- Illinois Institute of Technology, Department of Computer Science -----------------#
#---------------------------------------------------------------------------------------------#
# annotate_pos -------------------------------------------------------------------------------#
# INPUT: Plain text files, parameter settings ------------------------------------------------#
# OUTPUT: CoNLL formatted pos-tagged files, with emojis identified and labelled --------------#
#---------------------------------------------------------------------------------------------#

def annotate_pos(input_files, 
					emoji_file, 
					input_folder, 
					settings_dictionary, 
					encoding_type, 
					number_of_cpus_annotate, 
					docs_per_file):

	#Import required script-specific modules#
	from functions_annotate.annotate_files import annotate_files
	from functions_input.create_emoji_dictionary import create_emoji_dictionary
	import time

	start_beginning = time.time()
	#---------------------------------------------------------------------------------------------#
	#1: Annotate plain text input files  ---------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#
	
	emoji_dictionary = create_emoji_dictionary(emoji_file)
	
	
	for input_file in input_files:
		
		conll_files = annotate_files(input_folder, 
										input_file, 
										settings_dictionary, 
										encoding_type, 
										number_of_cpus_annotate, 
										emoji_dictionary, 
										docs_per_file
									)
									
	print("Finished tagging files.")
	
	return
#-------------------------------------------------------------------------------------------------#

#Prevent pool workers from starting here#
if __name__ == '__main__':
#---------------------------------------#

	#CODE FOR RUNNING FROM COMMAND LINE#
	import sys

	#Get parameters file to use#
	parameters_file = str(sys.argv[1])

	#Import parameter values and global variables#
	#All parameter and global variables must be prefaced with "pm."  ---#

	import importlib
	from annotate_pos import annotate_pos

	try:
		pm = importlib.import_module(parameters_file)
	except ImportError:
		print("Error in specified parameters file. Format is 'files_parameters.FILENAME'")
			
	annotate_pos(pm.input_files, 
					pm.emoji_file, 
					pm.input_folder, 
					pm.settings_dictionary, 
					pm.encoding_type, 
					pm.number_of_cpus_annotate, 
					pm.docs_per_file
					)
					
	#END CODE FOR RUNNING FROM COMMAND LINE#