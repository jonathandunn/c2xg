import pickle
import time
import os
import random

if __name__ == "__main__":

	from modules.Encoder import Encoder
	from modules.Loader import Loader
	from modules.Parser import Parser
	from modules.MDL_Learner import MDL_Learner

	#Load association measure module
	in_dir = os.path.join("..", "..", "..", "..", "Test", "In")
	out_dir = os.path.join("..", "..", "..", "..", "Test", "Out")
	Load = Loader(in_dir, out_dir, language = "eng")
	Encode = Encoder(Loader = Load)
	Parse = Parser(Load, Encode)
	
	#Initiate MDL
	MDL = MDL_Learner(Load, Encode, Parse, freq_threshold = 25)

	
	#Get MDL annotated data
	test_files = ["eng.test.txt"]
	MDL.get_mdl_data(test_files, workers = 10)
	
	#Delete after testing
	Load.save_file(MDL, "mdl.p")
	#MDL = Load.load_file("mdl.p")
	MDL.search(turn_limit = 10, workers = 16)
	
	#Evaluate a subset of the total candidates
	#total_mdl = MDL.evaluate_subset(subset)