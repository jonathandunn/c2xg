import pickle
import os

if __name__ == "__main__":

	from modules.Candidates import Candidates
	from modules.Loader import Loader
	
	#Set input and output paths
	in_dir = os.path.join("..", "..", "..", "..", "Test", "In")
	out_dir = os.path.join("..", "..", "..", "..", "Test", "Out")
	Load = Loader(in_dir, out_dir, language = "eng")
	
	Candidates = Candidates(language = "eng", Loader = Load)
	print("Initialized Candidates")
	
	#Start processing
	candidate_dict = Candidates.find(ngrams = (3,6), threshold = 10, save = True, workers = 10)
	#candidate_dict = Load.load_file("eng.candidates.merged.p")
	
	candidate_df = Candidates.get_association(candidate_dict, save = True)