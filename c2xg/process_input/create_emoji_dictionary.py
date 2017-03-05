#-------------------------------------------------------------------------------#
#INPUT: List of unicode sequences with emoji descriptions ----------------------#
#OUTPUT: Dictionary with UTF-8 string as keys and descriptions as values -------#
#-------------------------------------------------------------------------------#
def create_emoji_dictionary(emoji_file):

	import codecs
	
	emoji_dictionary = {}
	
	fo = codecs.open(emoji_file, "r", encoding = "utf-8")
	
	for line in fo:
	
		line = line.replace("\r","").replace("\n","")
		line_list = line.split("\t")
		
		current_symbol = line_list[0]
		current_label = " EMOJI" + str(line_list[1].upper()) + "EMOJI "
		
		emoji_dictionary[current_symbol] = current_label
		
	fo.close()
	
	emoji_list = list(emoji_dictionary.keys())
	emoji_list.sort(key = len, reverse = True)
	
	emoji_dictionary['emoji_list'] = emoji_list

	return emoji_dictionary
#--------------------------------------------------#