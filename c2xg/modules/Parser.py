from modules.Encoder import Encoder
from modules.Loader import Loader


class Parser(object):

	def __init__(self, Loader, Encoder, grammar):
	
		#Initialize Parser
		self.language = Encoder.language
		self.Encoder = Encoder
		self.Loader = Loader	
		self.grammar = grammar
	
	#--------------------------------------------------------------#
	
	def parse_stream(self, files):
	
		for line in self.Encoder.load_stream(files):
			line = self.parse(line)
			yield line
			
	#--------------------------------------------------------------#

	def parse(self, line):

		#Iterate over line from left to right
		for i in range(len(line)):
			
			unit = line[i]
			remaining = len(line) - i
			
			#Check for plausible candidates moving forward
			new_candidates = [construction for construction in self.grammar if construction[0][1] == unit[construction[0][0]-1] and remaining >= len(construction)]
			
			#Check existing candidates
			
			#Merge new and remaining candidates

		sys.kill
	