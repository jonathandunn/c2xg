from c2xg import C2xG
import os

if __name__ == "__main__":

	language = "eng"
	
	#Initialize C2xG object
	CxG = C2xG(data_dir = "../../../!Data", language = language)
		
	#Start or resume learning
	CxG.learn(nickname = language, 
				cycles = 1, 
				cycle_size = (1, 20, 100), 
				ngram_range = (3,6),
				freq_threshold = 200,
				turn_limit = 10,
				workers = 4,
				states = [(1, "Candidate_State", "None"), (1, "Candidate", ["eng.88.txt", "eng.89.txt"])]
				)
				
# import pickle
# import time
# import os
# import random

# if __name__ == "__main__":
	
	# from modules.Association import Association
	# from modules.Encoder import Encoder
	# from modules.Loader import Loader
	
	# #Set input and output paths
	# in_dir = os.path.join("..", "..", "..", "!Data", "In")
	# out_dir = os.path.join("..", "..", "..", "!Data", "Out")
	
	# #Initiate Loader and Association objects; all files in input_directory that end with ".txt" will be used
	# Load = Loader(in_dir, out_dir, language = "eng")
	# Association = Association(language = "eng", Loader = Load)

	# #Find ngrams, save results to files to support very large datasets
	# Association.find_ngrams(workers = 10)

	# #Merge ngrams
	# ngrams = Association.merge_ngrams()
	