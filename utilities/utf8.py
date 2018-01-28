import codecs

fo = open("Russian.Formatted.Testing.NoSent.txt", "rb")
fw = codecs.open("Russian.Formatted.Testing.NoSent.utf8", "w", encoding = "utf8")

for line in fo:
	line = line.decode("utf8", errors = "replace")
	fw.write(line)
	
fo.close()
fw.close()