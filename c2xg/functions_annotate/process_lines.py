#-------------------------------------------------------------------------------#
#INPUT: Line tuple (ID, Str) and Emoji dictionary ------------------------------#
#OUTPUT: Tokenized line tuple with emojis identified and labelled --------------#
#-------------------------------------------------------------------------------#
def process_lines(line_tuple, emoji_dictionary):

	from functions_annotate.tokenize_line import tokenize_line
	from functions_annotate.replace_emojis import replace_emojis
	
	id = line_tuple[0]
	line = line_tuple[1]
	
	if line:
	
		line = line.replace("\r", "").replace("\n","")
	
		current_line = replace_emojis(line, emoji_dictionary)
		current_line = tokenize_line(current_line)
	
		return (id, current_line)
		
	else:
		
		return
#--------------------------------------------------#