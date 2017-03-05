#---------------------------------------------------------------------#
def save_mwes(final_grammar, Parameters, Grammar):

	from candidate_extraction.write_candidates import write_candidates
	
	#Saving MWEs only requires writing lists of the lexical items#
	mwe_list = list(final_grammar)
	mwe_grammar_list = []
		
	for mwe in mwe_list:
			
		mwe = eval(mwe)
		mwe_sequence = ""
			
		for pair in mwe:
				
			mwe_sequence += str(Grammar.Lemma_List[pair[1]])
			mwe_sequence += str(" ")
				
		mwe_grammar_list.append(mwe_sequence)
		
	#Now save and write MWE grammar#
	Grammar.MWE_List = mwe_grammar_list
	write_candidates(Parameters.Data_File_MWEs, Grammar)
		
	#Write to debug file if necessary#\
	if Parameters.Debug == True:
		
		import codecs
		fw = codecs.open(Parameters.Debug_File + "Grammar.MWE", "w", encoding = Parameters.Encoding_Type, errors = "replace")
			
		for construction in mwe_grammar_list:
			fw.write(str(construction) + "\n")
	
	return Grammar
#------------------------------------------------------------------------------#