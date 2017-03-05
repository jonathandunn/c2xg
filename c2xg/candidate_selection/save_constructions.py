#---------------------------------------------------------------------#
def save_constructions(final_grammar, Parameters, Grammar):

	from candidate_extraction.write_candidates import write_candidates 
	
	#Now save and write grammar#
	Grammar.Construction_List = final_grammar
	write_candidates(Parameters.Data_File_Constructions, Grammar)
		
	#Write to debug file if necessary#\
	if Parameters.Debug == True:
		
		import codecs
		fw = codecs.open(Parameters.Debug_File + "Grammar." + "Constructions", "w", encoding = Parameters.Encoding_Type, errors = "replace")
			
		for construction in final_grammar:
			fw.write(str(construction) + "\n")
	
	return
#------------------------------------------------------------------------------#