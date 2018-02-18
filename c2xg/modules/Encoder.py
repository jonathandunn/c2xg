import os
import pickle
import re
import cytoolz as ct
from gensim.parsing import preprocessing
from modules.rdrpos_tagger.Utility.Utils import readDictionary
from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import RDRPOSTagger
from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import unwrap_self_RDRPOSTagger
from modules.rdrpos_tagger.pSCRDRtagger.RDRPOSTagger import printHelp
from sklearn.utils import murmurhash3_32

#Fix RDRPos import
current_dir = os.getcwd()
if current_dir == "Utility":
	os.chdir(os.path.join("..", "..", ".."))

#Changes the generation of lexicon / dictionary used
DICT_CONSTANT = ".DIM=500.SG=1.HS=1.ITER=25.p"
#-------------------------------------------------------------------------------------------#

class Encoder(object):

	#---------------------------------------------------------------------------#
	def __init__(self, language, Loader, word_classes = False):
		
		self.language = language
		self.Loader = Loader

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
		pos_list = ["PROPN", "SYM", "VERB", "DET", "CCONJ", "AUX", "ADJ", "INTJ", "SCONJ", "PRON", "NUM", "PUNCT", "ADV", "ADP", "X", "NOUN", "PART"]
		self.pos_dict = {murmurhash3_32(pos, seed=0): pos for pos in pos_list}
		
		#Get semantic dict
		if word_classes == False:
			dictionary_file = os.path.join(".", "data", "dictionaries", language + DICT_CONSTANT)
			with open(dictionary_file, "rb") as fo:
				self.word_dict = pickle.load(fo)
			
			#UPDATE ONCE HAVE NEW DICTSs
			self.domain_dict = {murmurhash3_32(key, seed=0): self.word_dict[key]["domain"] for key in self.word_dict.keys()}
			self.word_dict = {murmurhash3_32(key, seed=0): key for key in self.word_dict.keys()}
			
			self.build_decoder()

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
	
		sequence = [self.decoding_dict[pair[0]][pair[1]] for pair in item]
			
		return " ".join(sequence)		

	#---------------------------------------------------------------------------#
	
	def load_stream(self, input_files, word_classes = False):	
	
		for file in input_files:
			for line in self.load(file, word_classes = word_classes):
				yield line
		
	#---------------------------------------------------------------------------#
		
	def load(self, file, word_classes = False):

		#zho needs an additional tokenizer
		if self.language == "zho":
			
			import modules.jieba.jeiba as jb
			tk = jb.Tokenizer()
			tk.initialize()
			tk.lock = True
				
		for line in self.Loader.read_file(file):
			
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
			
			if word_classes == False:
				line = self.r.tagRawSentenceHash(rawLine = line, DICT = self.DICT, word_dict = self.domain_dict)
				#Array of tuples (LEX, POS, CAT)
				
			#For training word embeddings, just return the list
			else:
				line = self.r.tagRawSentenceGenSim(rawLine = line, DICT = self.DICT)

			yield line