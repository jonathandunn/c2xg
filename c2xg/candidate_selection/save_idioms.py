#---------------------------------------------------------------------#
def save_idioms(final_grammar, Parameters, Grammar):

	from candidate_selection.write_grammar_debug import write_grammar_debug
	from candidate_extraction.write_candidates import write_candidates
	from process_input.rdrpos_tagger.Utility.Utils import readDictionary
				
	import os
	import codecs
	import platform
	
	from process_input.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger
	from process_input.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import unwrap_self_RDRPOSTagger
	from process_input.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import printHelp
	
	r = RDRPOSTagger()
			
	#Check and Change directory if necessary; only once if multi-processing#
	current_dir = os.getcwd()

	if platform.system() == "Windows":
		slash_index = current_dir.rfind("\\")
				
	else:
		slash_index = current_dir.rfind("/")
				
	current_dir = current_dir[slash_index+1:]
			
	if current_dir == "Utility":
		os.chdir("../../../")
	#End directory check#
			
	model_string = "./files_data/pos_rdr/" + Parameters.Language + ".RDR"
	dict_string = "./files_data/pos_rdr/"  + Parameters.Language + ".DICT"
		
	r.constructSCRDRtreeFromRDRfile(model_string)
	DICT = readDictionary(dict_string)
	
	#Now tag and process each idiom#
	idiom_list = []
	sequence_list = []
	
	for sequence in final_grammar:
	
		idiom = ""
		
		for unit in sequence:
			idiom += str(Grammar.Lemma_List[unit[1]]) + " "
		
		idiom_annotated = r.tagRawSentence(DICT, idiom)
		idiom_annotated = idiom_annotated.split()

		word_list = []
		past_tag = ""
		bad_flag = 0
		
		for pair in idiom_annotated:

			pair = pair.split("/")
			word = pair[0] + " "
			tag = pair[1]
			
			word_list.append(str(word))
			
			if past_tag != "":
				if tag != past_tag:
					bad_flag = 1
					
			past_tag = tag
		
		if bad_flag == 0:

			idiom_string = "".join(word_list)
			idiom_string = idiom_string[0:len(idiom_string) -1]
			
			idiom_tagged = idiom_string.replace(" ", "_")
			
			idiom_list.append([idiom_string, idiom_tagged, tag])
			sequence_list.append(sequence)
		
	print(idiom_list)
		
	Grammar.Idiom_List = idiom_list
	
	write_grammar_debug(sequence_list, "Idioms", Grammar, Parameters)
	write_candidates(Parameters.Data_File_Idioms, Grammar)
	
	return Grammar
#------------------------------------------------------------------------------#