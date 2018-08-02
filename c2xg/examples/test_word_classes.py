import os
from modules.Word_Classes import Word_Classes
from modules.Loader import Loader

if __name__ == "__main__":

	#Set input and output paths
	in_dir = os.path.join("..", "..", "..", "..", "Test", "In")
	out_dir = os.path.join("..", "..", "..", "..", "Test", "Out")
		
	#Initiate Loader and Word_Classes objects; all files in input_directory that end with ".txt" will be used
	Load = Loader(in_dir, out_dir, language = "por")
	Classes = Word_Classes(Loader = Load)

	#Train model
	#model_file = Classes.train(size = 500, min_count = 500, workers = 12)

	#Or, load instead of training
	model_file = Load.load_file("por.POS.2000dim.2000min.20iter.Vectors.p")
	nickname = "por.POS.2000dim.2000min.20iter.Vectors.Clusters"

	#Cluster embeddings
	mixed_classes, pos_classes = Classes.build_clusters(model_file, nickname, workers = 8)

	#Write cluster dictionary
	Classes.write_clusters(mixed_classes, nickname, "Mixed")
	Classes.write_clusters(pos_classes, nickname, "POS")

