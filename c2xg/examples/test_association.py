import os
import pickle

if __name__ == "__main__":

	from modules.Association import Association
	from modules.Encoder import Encoder
	from modules.Loader import Loader
	
	#Set input and output paths
	in_dir = os.path.join("..", "..", "..", "..", "Test", "In")
	out_dir = os.path.join("..", "..", "..", "..", "Test", "Out")
	
	#Initiate Loader and Association objects; all files in input_directory that end with ".txt" will be used
	Load = Loader(in_dir, out_dir, language = "eng")
	Association = Association(language = "eng", Loader = Load)

	#Find ngrams, save results to files to support very large datasets
	Association.find_ngrams(workers = 10)

	#Merge ngrams
	ngrams = Association.merge_ngrams()

	#Calculate pairwise association		
	association_dict = Association.calculate_association(ngrams, save = True)		
	
	#Check association values
	Encoder = Encoder(Load)

	#Decode top pairs from the get_top generator
	for pair, value in Association.get_top(association_dict, "LR", 10):
		print(Encoder.decode(pair), value)
	
	#Decode top pairs in the other direction of association
	for pair, value in Association.get_top(association_dict, "RL", 10):
		print(Encoder.decode(pair), value)
	
	#Load.clean()