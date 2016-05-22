#---------------------------------------------#
#Take string, return with emojis labelled ----#
#---------------------------------------------#
def replace_emojis(line, emoji_dictionary):
	
	emoji_list = emoji_dictionary['emoji_list']
	
	for emoji in emoji_list:
		
		if emoji in line:
			line = line.replace(emoji, " " + emoji_dictionary[emoji] + " ")
	
	return line
#---------------------------------------------#