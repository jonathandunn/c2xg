#---------------------------------------------#
#Take string, return with emojis labelled ----#
#---------------------------------------------#
def replace_emojis(line, Parameters):
	
	emoji_list = Parameters.Emoji_Dictionary['emoji_list']
	
	for emoji in emoji_list:
		
		if emoji in line:
			line = line.replace(emoji, " " + Parameters.Emoji_Dictionary[emoji] + " ")
	
	return line
#---------------------------------------------#