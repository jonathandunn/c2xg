import os
import pickle
import re
from pathlib import Path
import collections
import cytoolz as ct
import numpy as np
import pandas as pd
from cleantext import clean
from sklearn.utils import murmurhash3_32

#Changes the generation of lexicon / dictionary used
DICT_CONSTANT = ".clusters.fastText.v2.gz"

from .rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger
from .rdrpos_tagger.Utility.Utils import getWordTag, getRawText, readDictionary
from .rdrpos_tagger.InitialTagger.InitialTagger import initializeCorpus, initializeSentence

#-------------------------------------------------------------------------------------------#

class Encoder(object):

	#---------------------------------------------------------------------------#
	def __init__(self, Loader, word_classes = False, zho_split = False):
		
		self.language = Loader.language
		self.zho_split = zho_split
		self.Loader = Loader
		
		MODEL_STRING = Path(__file__).parent / os.path.join("..", "data", "pos_rdr", self.language + ".RDR")
		DICT_STRING = Path(__file__).parent / os.path.join("..", "data", "pos_rdr", self.language + ".DICT")
		DICTIONARY_FILE = Path(__file__).parent / os.path.join("..", "data", "dictionaries", self.language + DICT_CONSTANT)

		#zho needs an additional tokenizer
		if self.language == "zho":
			
			try:
				import modules.jieba.jeiba as jb
			except:
				import c2xg.modules.jieba.jeiba as jb
				
			self.tk = jb.Tokenizer()
			self.tk.initialize()
			self.tk.lock = True

		#Universal POS Tags are fixed across languages
		pos_list = ["PROPN", "SYM", "VERB", "DET", "CCONJ", "AUX", "ADJ", "INTJ", "SCONJ", "PRON", "NUM", "PUNCT", "ADV", "ADP", "X", "NOUN", "PART"]
		self.pos_dict = {murmurhash3_32(pos, seed=0): pos for pos in pos_list}
		
		#Get semantic dict
		self.word_dict = pd.read_csv(DICTIONARY_FILE, index_col = 0).to_dict()["Cluster"]
		self.domain_dict = {murmurhash3_32(str(key), seed=0): self.word_dict[key] for key in self.word_dict.keys()}
		self.word_dict = {murmurhash3_32(str(key), seed=0): key for key in self.word_dict.keys()}

		#Build decoder
		self.build_decoder()

		#Initialize tagger
		self.DICT = readDictionary(DICT_STRING)
		self.r = RDRPOSTagger(word_dict = self.domain_dict, DICT = self.DICT)
		self.r.constructSCRDRtreeFromRDRfile(MODEL_STRING)

	#-----------------------------------------------------------------------------------#
		
	def build_decoder(self):

		#Create a decoding resource
		#LEX = 1, POS = 2, CAT = 3
		decoding_dict = {}
		decoding_dict[1] = self.word_dict
		decoding_dict[2] = self.pos_dict
		decoding_dict[3] = {key: "<" + str(key) + ">" for key in list(set(self.domain_dict.values()))}
			
		self.decoding_dict = decoding_dict

	#---------------------------------------------------------------------------#
	
	def decode(self, item):
	
		sequence = [self.decoding_dict.get([pair[0]][pair[1]], "UNK") for pair in item]
			
		return " ".join(sequence)		

	#---------------------------------------------------------------------------#
	
	def decode_construction(self, item):
		
		sequence = []
		for pair in item:

			try:
				val = self.decoding_dict[pair[0]][pair[1]]
			except:
				val = "UNK"

			sequence.append(val)
			
		return "[ " + " -- ".join(sequence) + " ]"

	#---------------------------------------------------------------------------#
	
	def load_stream(self, input_files, no_file = False):

		if no_file == False:

			#If only got one file, wrap in list
			if isinstance(input_files, str):
				input_files = [input_files]
		
			for file in input_files:
				for line in self.Loader.read_file(file):
					if len(line) > 1:
						line = self.load(line)
						yield line

		elif no_file == True:
			for line in input_files:
					if len(line) > 1:
						line = self.load(line)
						yield line
					
	#---------------------------------------------------------------------------#

	def load_examples(self, input_files, word_classes = False):

		#If only got one file, wrap in list
		if isinstance(input_files, str):
			input_files = [input_files]
	
		for file in input_files:
			for line in self.Loader.read_file(file):
				if len(line) > 1:
					line2 = self.load(line)
					yield line, line2

	#---------------------------------------------------------------------------#
	
	def load_batch(self, input_lines, word_classes = False):	
	
		return_list = []	#Initiate holding list
		
		for line in input_lines:
			if len(line) > 1:
				return_list.append(self.load(line))
		
		return return_list
		
	#---------------------------------------------------------------------------#
		
	def load(self, line):

		#Tokenize zho
		if self.language == "zho" and self.zho_split == True:
								
			line = [x for x in self.tk.cut(line, cut_all = True, HMM = True) if x != ""]
			line = " ".join(line)

		#Use clean-text
		line = clean(line,
						fix_unicode = True,
						to_ascii = False,
						lower = True,
						no_line_breaks = True,
						no_urls = True,
						no_emails = True,
						no_phone_numbers = True,
						no_numbers = True,
						no_digits = True,
						no_currency_symbols = True,
						no_punct = True,
						replace_with_punct = "",
						replace_with_url = "<URL>",
						replace_with_email = "<EMAIL>",
						replace_with_phone_number = "<PHONE>",
						replace_with_number = "<NUMBER>",
						replace_with_digit = "0",
						replace_with_currency_symbol = "<CUR>"
						)

		line = self.r.tagRawSentenceHash(rawLine = line)
		#Array of tuples (LEX, POS, CAT)

		return np.array(line)