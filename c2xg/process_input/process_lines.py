#-------------------------------------------------------------------------------#
#INPUT: Line tuple (ID, Str) and Emoji dictionary ------------------------------#
#OUTPUT: Tokenized line tuple with emojis identified and labelled --------------#
#-------------------------------------------------------------------------------#
def process_lines(line_tuple, Parameters, Grammar):

	from process_input.tokenize_line import tokenize_line
	from process_input.replace_emojis import replace_emojis
	from process_input.find_mwes import find_mwes
	
	id = line_tuple[0]
	line = line_tuple[1]
	
	if line and line != "" and line != None:
	
		line = line.replace("\r", "").replace("\n","")
	
		current_line = replace_emojis(line, Parameters)
		current_line = tokenize_line(current_line)
		
		# #Only find MWEs if the MWE Grammar has been passed#
		# if Grammar != None:
			# print(Grammar.MWE_List)
			# sys.kill()
			# if Grammar.Type == "MWE":
				# current_line = find_mwes(current_line, Grammar)

		return (id, current_line)
		
	else:
		
		return
#--------------------------------------------------#