import os
import os.path
import pickle
import codecs
import time
import re
import cytoolz as ct
import pandas as pd
from gensim.parsing import preprocessing
from modules.rdrpos_tagger.Utility.Utils import readDictionary
from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger
from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import unwrap_self_RDRPOSTagger
from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import printHelp

#Fix RDRPos import
current_dir = os.getcwd()
if current_dir == "Utility":
	os.chdir(os.path.join("..", "..", ".."))

#Changes the generation of lexicon / dictionary used
DICT_CONSTANT = ".DIM=500.SG=1.HS=1.ITER=25.p"
#-------------------------------------------------------------------------------------------#

class Encoder(object):

	#---------------------------------------------------------------------------#
	def __init__(self, language):
		
		self.language = language

		#Initialize RDRPosTagger
		model_string = os.path.join(".", "data", "pos_rdr", self.language + ".RDR")
		dict_string = os.path.join(".", "data", "pos_rdr", self.language + ".DICT")
				
		#Initialize tagger
		self.r = RDRPOSTagger()
		self.r.constructSCRDRtreeFromRDRfile(model_string) 
		self.DICT = readDictionary(dict_string) 
				
		# #Initialize emoji remover
		try:
		# Wide UCS-4 build
			self.myre = re.compile(u'['
				u'\U0001F300-\U0001F64F'
				u'\U0001F680-\U0001F6FF'
				u'\u2600-\u26FF\u2700-\u27BF]+', 
				re.UNICODE)
		except re.error:
			# Narrow UCS-2 build
				self.myre = re.compile(u'('
				u'\ud83c[\udf00-\udfff]|'
				u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
				u'[\u2600-\u26FF\u2700-\u27BF])+', 
				re.UNICODE)
		
		#Universal POS Tags are fixed across languages
		self.pos_dict = {}
		self.pos_dict['propn'] = 1
		self.pos_dict['sym'] = 2
		self.pos_dict['verb'] = 3
		self.pos_dict['det'] = 4
		self.pos_dict['cconj'] = 5
		self.pos_dict['aux'] = 6
		self.pos_dict['adj'] = 7
		self.pos_dict['sconj'] = 8
		self.pos_dict['pron'] = 9
		self.pos_dict['num'] = 10
		self.pos_dict['punct'] = 11
		self.pos_dict['adv'] = 12
		self.pos_dict['adp'] = 13
		self.pos_dict['x'] = 14
		self.pos_dict['noun'] = 15
		self.pos_dict['part'] = 16
		self.pos_dict['intj'] = 17
		
		#Get semantic dict
		dictionary_file = os.path.join(".", "data", "dictionaries", language + DICT_CONSTANT)
		with open(dictionary_file, "rb") as fo:
			self.word_dict = pickle.load(fo)

		
	def build_decoder():
	
		#Create a decoding resource
		decoding_dict = {}
		decoding_dict[0] = {}
		decoding_dict[1] = {}
		decoding_dict[2] = {}
		
		for pos in self.pos_dict.keys():
			current_index = self.pos_dict[pos]
			decoding_dict[2][current_index] = pos.upper()
			
		for word in self.word_dict.keys():
			current_index = self.word_dict[word]["index"]
			current_domain = self.word_dict[word]["domain"]
			
			decoding_dict[1][current_index] = word
			decoding_dict[3][current_domain] = "<" + str(current_domain) + ">"
			
		decoding_dict[1][0] = "NONE"
		decoding_dict[2][0] = "NONE"
		decoding_dict[3][0] = "NONE"
		
		self.decoding_dict = decoding_dict

	#---------------------------------------------------------------------------#
	def decode(self, item):
	
		sequence = [self.decoding_dict[pair[0]][pair[1]] for pair in item]
			
		return " ".join(sequence)		

	#---------------------------------------------------------------------------#
	def load(self, files):

		#Check if getting a file or list of files
		if not isinstance(files, list):
			files = [files]
		
		#zho needs an additional tokenizer
		if self.language == "zho":
			
			import modules.jieba.jeiba as jb
			tk = jb.Tokenizer()
			tk.initialize()
			tk.lock = True
				
		for fname in files:
			
			#Yield sentence annotations one sentence at a time
			with codecs.open(fname, encoding = "utf-8", errors = "replace") as fo:
				for line in fo:
										
					#Tokenize zho
					if self.language == "zho":
								
						line = [x for x in tk.cut(line, cut_all = True, HMM = True) if x != ""]
						line = " ".join(line)


					#Remove links, hashtags, at-mentions, mark-up, and "RT"
					line = re.sub(r"http\S+", "", line)
					line = re.sub(r"@\S+", "", line)
					line = re.sub(r"#\S+", "", line)
					line = re.sub("<[^>]*>", "", line)
					line = line.replace(" RT", "").replace("RT ", "")
								
					#Remove emojis
					line = re.sub(self.myre, "", line)
									
					#Remove punctuation and extra spaces
					line = ct.pipe(line, 
									preprocessing.strip_tags, 
									preprocessing.strip_punctuation, 
									preprocessing.split_alphanum,
									preprocessing.strip_non_alphanum,
									preprocessing.strip_multiple_whitespaces
									)
									
					#Strip and reduce to max training length
					line = line.lower().strip().lstrip()
					line = self.r.tagRawSentence(rawLine = line, DICT = self.DICT, word_dict = self.word_dict, pos_dict = self.pos_dict)
					#Array of tuples (LEX, POS, CAT)

					yield line