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
	Encode = Encoder(Loader = Load)
	
	#Load candidate constructions to use as grammar
	candidates = Load.load_file("eng.candidates.merged.p")
	candidates = list(candidates.keys())
	
	#Load parsing module
	Parse = Parser(Load, Encode)
	grammar = Parse.format_grammar(candidates)		#Reformat candidate to be equal length for numba
		
	files = ["eng.1.txt"]
	
	# starting = time.time()
	# lines = Parse.parse_prep(files, workers = 10)	#No need to reencode the test set many times
	# print("\tLoaded and encoded " + str(len(lines)) + " words in " + str(time.time() - starting))
	
	# starting = time.time()
	# results = Parse.parse_batch_mdl(lines, grammar, workers = 10)
	# print("\tParsed " + str(len(lines)) + " words with " + str(len(grammar)) + " constructions in " + str(time.time() - starting) + " seconds.")
	
	# print(results.keys())
	
	#-------------------------------
	#Streaming version for feature extraction
	grammar = grammar[0:10000]
	starting = time.time()
	counter = 0
	
	for matches in Parse.parse_stream(files, grammar):
		counter += 1
	
	print("Time: " + str(time.time() - starting))
	print("Sentences: " + str(counter))
	print("Constructions: " + str(len(grammar)))