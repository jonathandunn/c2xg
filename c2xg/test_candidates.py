import pickle
import os

if __name__ == "__main__":

	from modules.Candidates import Candidates
	from modules.Loader import Loader
	
	#Set input and output paths
	in_dir = os.path.join("..", "..", "..", "..", "!Corpora", "!Background Data", "eng")
	out_dir = os.path.join("..", "..", "..", "..", "data", "Test")
	Load = Loader(in_dir, out_dir)
	
	Candidates = Candidates(language = "eng", Loader = Load)
	
	#Start processing
	candidate_dict = Candidates.run(ngrams = (3,6), save = True, workers = 4)