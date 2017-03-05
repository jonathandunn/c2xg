#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

# -- Wrapper function for pos-tagging, tokenizing, recognizing emojis, and writing CONLL formatted input files

def annotate_pos(Parameters, Grammar = ""):

	print("")
	print("Starting C2xG.Annotate_POS")
	print("")
	
	#Import required script-specific modules#
	from process_input.annotate_files import annotate_files
	import time

	start_beginning = time.time()
	#---------------------------------------------------------------------------------------------#
	#1: Annotate plain text input files  ---------------------------------------------------------#
	#---------------------------------------------------------------------------------------------#
	
	for input_file in Parameters.Input_Files:
		
		conll_files = annotate_files(input_file, Parameters, Grammar)
									
	Parameters.Run_Tagger = False
	Parameters.Input_Files = conll_files
	
	print("Finished tagging files.")
	
	return
#-------------------------------------------------------------------------------------------------#