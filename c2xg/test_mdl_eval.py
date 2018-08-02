import pickle
import time
import os
import random

if __name__ == "__main__":
	
	from c2xg import C2xG
	from modules.Encoder import Encoder
	from modules.Loader import Loader
	from modules.Parser import Parser
	from modules.MDL_Learner import MDL_Learner

	#Load association measure module
	in_dir = os.path.join("..", "..", "cxg_data")
	CxG = C2xG(data_dir = in_dir, language = "eng")
	
	print(CxG.language)
	print(CxG.n_features)
	
	#Get MDL annotated data
	test_files = ["eng.10.txt", "eng.20.txt", "eng.30.txt", "eng.40.txt", "eng.50.txt"]
	results = CxG.eval_mdl(test_files, workers = 32)
	
