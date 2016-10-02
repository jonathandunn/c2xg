#-----C2xG, v 1.0 ----------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
#---- Copyright, 2015-2016 Jonathan E. Dunn --------------------------------------------------#
#---------- www.jdunn.name -------------------------------------------------------------------#
#---------- jonathan.edwin.dunn@gmail.com ----------------------------------------------------#
#---------- Illinois Institute of Technology, Department of Computer Science -----------------#
#---------------------------------------------------------------------------------------------#

def learn_dictionary(nickname, 
						input_directory, 
						output_directory, 
						min_threshold, 
						num_dimensions, 
						num_clusters,
						sg,
						number_of_cpus = 1
						):

	from functions_semantic_dictionary.train_models import train_models
	from functions_semantic_dictionary.write_clusters import write_clusters
	from functions_semantic_dictionary.build_clusters import build_clusters
	
	model_file = nickname + ".Model.p"
	cluster_file = nickname + ".Clusters.p"
	output_file = nickname + ".Dictionary.txt"

	print("Starting " + nickname + " on files in " + input_directory)

	train_models(input_directory, number_of_cpus, min_threshold, num_dimensions, model_file, sg)
	build_clusters(model_file, num_clusters, cluster_file)
	write_clusters(cluster_file, num_clusters, output_file)
	
	return
#--------------------------------------------------------------------------------------------#

nickname = "German.Aranea"
input_directory = "./Testing/"
output_directory = "./files_data"
num_workers = 12
min_threshold = 10
num_dimensions = 400
num_clusters = 100
sg = 1

learn_dictionary(nickname, 
					input_directory, 
					output_directory, 
					min_threshold, 
					num_dimensions, 
					num_clusters,
					num_workers,
					sg
					)