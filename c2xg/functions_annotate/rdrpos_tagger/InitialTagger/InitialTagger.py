# -*- coding: utf-8 -*-

import re

def initializeSentence(FREQDICT, sentence):
	words = sentence.strip().split()
	taggedSen = []
	for word in words:
		if word in ["“", "”", "\""]:
			taggedSen.append("''/" + FREQDICT["''"])
			continue
		
		tag = ''
		decodedW = word
		lowerW = decodedW.lower()
		if word in FREQDICT:
			tag = FREQDICT[word]
		elif lowerW in FREQDICT:
			tag = FREQDICT[lowerW]
		else:
			if re.search(r"[0-9]+", word) != None:
				tag = FREQDICT["TAG4UNKN-NUM"]
			else:
				suffixL2 = suffixL3 = suffixL4 = suffixL5 = None
				wLength = len(decodedW)
				if wLength >= 4:
					suffixL3 = ".*" + decodedW[-3:]
					suffixL2 = ".*" + decodedW[-2:]
				if wLength >= 5:
					suffixL4 = ".*" + decodedW[-4:]
				if wLength >= 6:
					suffixL5 = ".*" + decodedW[-5:]
				
				if suffixL5 in FREQDICT:
					tag = FREQDICT[suffixL5]
				elif suffixL4 in FREQDICT:
					tag = FREQDICT[suffixL4] 
				elif suffixL3 in FREQDICT:
					tag = FREQDICT[suffixL3]
				elif suffixL2 in FREQDICT:
					tag = FREQDICT[suffixL2]
				elif decodedW[0].isupper():
					tag = FREQDICT["TAG4UNKN-CAPITAL"]
				else:
					tag = FREQDICT["TAG4UNKN-WORD"]
		   
		taggedSen.append(word + "/" + tag)								
	
	return " ".join(taggedSen)

def initializeCorpus(FREQDICT, inputFile, outputFile):
	import codecs
	lines = codecs.open(inputFile, "r", encoding = "utf-8").readlines()
	fileOut = codecs.open(outputFile, "w", encoding = "utf-8")
	for line in lines:
		fileOut.write(initializeSentence(FREQDICT, line) + "\n")
	fileOut.close()
