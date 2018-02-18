import os
from modules.Word_Classes import Word_Classes
from modules.Loader import Loader

#Set input and output paths
in_dir = os.path.join("..", "..", "..", "..", "..", "Test", "IN")
out_dir = os.path.join("..", "..", "..", "..", "..", "Test", "OUT")
	
#Initiate Loader and Word_Classes objects; all files in input_directory that end with ".txt" will be used
Load = Loader(in_dir, out_dir)
Classes = Word_Classes(language = "eng", Loader = Load)

#Train model
#model_file = Classes.train(size = 500, min_count = 500, workers = 12)

#Or, load instead of training
model_file = Load.load_file("eng.POS.500dim.500.min.Vectors.Complete.p")

#Cluster embeddings
mixed_classes, pos_classes = Classes.build_clusters(model_file, "eng.POS.500dim.500.min.Vectors.Complete")

#Write cluster dictionary
Classes.write_clusters(mixed_classes, "Mixed")
Classes.write_clusters(pos_classes, "POS")

