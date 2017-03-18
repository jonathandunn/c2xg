#---------------------------------------------#
#Take string, return a tokenized version -----#
#---------------------------------------------#
def tokenize_line(line):
	
	line = line.replace('"',' " ').replace(']',' ] ').replace('[',' [ ').replace(',',' , ').replace('--',' -- ')
	line = line.replace(')',' ) ').replace('(',' ( ').replace('>',' > ').replace('<',' < ')
	line = line.replace('&',' & ').replace('. ',' . ').replace("' "," ' ").replace('/',' / ')
	line = line.replace('“',' “ ').replace('’ ',' ’ ').replace('^',' ^ ').replace('\0','')
	line = line.replace('*',' * ').replace(': ',' : ').replace('@ ', '@').replace("# ","#")
	line = line.replace('+',' + ').replace('=',' = ').replace('~',' ~ ').replace('?',' ? ')
	line = line.replace('!',' ! ').replace('` ',' ` ').replace('”',' ” ').replace('\n','')
	line = line.replace('` ',' ` ').replace('…',' … ').replace('·',' · ').replace(';',' ; ')
	line = line.replace('\r', '').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace("... ", " ... ").replace("...", " ... ")
	
	try:
		if line[0] == " ":
			line = line[1:]
	
		return line
	
	except:
		return
#---------------------------------------------#