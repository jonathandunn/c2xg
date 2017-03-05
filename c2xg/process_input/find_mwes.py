#---------------------------------------#
def find_mwes(current_line, Grammar):

	current_line = current_line.lower()
	print(Grammar.MWE_List)

	for mwe in Grammar.MWE_List:
	
		try:
			current_line.replace(mwe, mwe.replace(" ", "_", 1))
			print(current_line)
		except:
			counter = None

	return current_line
#---------------------------------------#