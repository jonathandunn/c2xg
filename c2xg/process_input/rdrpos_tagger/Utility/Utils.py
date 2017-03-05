# -*- coding: utf-8 -*-

def getWordTag(wordTag):
	if wordTag == "///":
		return "/", "/"
	index = wordTag.rfind("/")
	word = wordTag[:index].strip()
	tag = wordTag[index + 1:].strip()
	return word, tag

def getRawText(inputFile, outFile):
	import codecs
	out = codecs.open(outFile, "w", encoding = "utf-8")
	sents = codecs.open(inputFile, "r", encoding = "utf-8").readlines()
	for sent in sents:
		wordTags = sent.strip().split()
		for wordTag in wordTags:
			word, tag = getWordTag(wordTag)
			if word != "":
				out.write(word + " ")
		out.write("\n")
	out.close()
	
def readDictionary(inputFile):
	import codecs
	dictionary = {}
	lines = codecs.open(inputFile, "r", encoding = "utf-8").readlines()
	for line in lines:
		wordtag = line.strip().split()
		dictionary[wordtag[0]] = wordtag[1]
	return dictionary

