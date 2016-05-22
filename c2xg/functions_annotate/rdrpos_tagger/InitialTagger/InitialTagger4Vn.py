# -*- coding: utf-8 -*-

import re

def isAbbre(word):

    word = str(word)
    for i in range(len(word)):
        if isVnLowerChar(word[i]) or word[i] == "_":
            return False
    return True

VNUPPERCHARS = ['Ă', 'Â', 'Đ', 'Ê', 'Ô', 'Ơ', 'Ư']
VNLOWERCHARS = ['ă', 'â', 'đ', 'ê', 'ô', 'ơ', 'ư']

def isVnLowerChar(char):
    if char.islower() or char in VNLOWERCHARS:
        return True;
    return False;

def isVnUpperChar(char):
    if char.isupper() or char in VNUPPERCHARS:
        return True;
    return False;

def isVnProperNoun(word):
    word = str(word)
    if (isVnUpperChar(word[0])):
        if word.count("_") >= 4:
            return True
        index = word.find("_")
        while index > 0 and index < len(word) - 1:
            if isVnLowerChar(word[index + 1]):
                return False;
            index = word.find("_", index + 1)
        return True;
    else:
        return False;

def initializeVnSentence(FREQDICT, sentence):
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
                if (re.search(r"[0-9]+", word) != None):
                    tag = FREQDICT["TAG4UNKN-NUM"]
                elif(len(word) == 1 and isVnUpperChar(word[0])):
                    tag = "Y"
                elif (isAbbre(word)):
                    tag = "Ny"
                elif (isVnProperNoun(word)):
                    tag = "Np"
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
                    else:
                        tag = FREQDICT["TAG4UNKN-WORD"]
                           
        taggedSen.append(word + "/" + tag) 
                                        
    return " ".join(taggedSen)

def initializeVnCorpus(FREQDICT, inputFile, outputFile):
    import codecs
    lines = codecs.open(inputFile, "r", encoding = "utf-8").readlines()
    fileOut = codecs.open(outputFile, "w", encoding = "utf-8")
    for line in lines:
        fileOut.write(initializeVnSentence(FREQDICT, line) + "\n")
    fileOut.close()

