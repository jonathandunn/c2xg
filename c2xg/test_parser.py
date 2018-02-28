import pickle
import time
import os

if __name__ == "__main__":

	from modules.Encoder import Encoder
	from modules.Loader import Loader
	from modules.Parser import Parser

	#Load association measure module
	in_dir = os.path.join("..", "..", "..", "..", "Test", "In")
	out_dir = os.path.join("..", "..", "..", "..", "Test", "Out")
	Load = Loader(in_dir, out_dir, language = "eng")
	Encode = Encoder(language = "eng", Loader = Load)
	
	#Load candidate constructions to use as grammar
	candidates = Load.load_file("eng.candidates.merged.p")
	candidates = list(candidates.keys())
	
	#Load parsing module
	Parse = Parser(Load, Encode, grammar = candidates)
	
	
	
	files = ["eng.1.txt"]
	counter = 0
		
	for line in Parse.parse_stream(files):
		
		counter += 1
		
		if counter == 10:
			sys.kill()