#-- C2xG, v 0.2
#-- Copyright, 2015-2017 Jonathan E. Dunn
#-- GNU LGPLv3
#-- www.jdunn.name
#-- jdunn8@iit.edu
#-- Illinois Institute of Technology, Department of Computer Science

#-- Wrapper for using GenSim's word2vec to learn lexical domains

def learn_dictionary(settings_dict, input_directory, output_directory, num_clusters, num_workers):

	from functions_semantic_dictionary.train_models import train_models
	from functions_semantic_dictionary.write_clusters import write_clusters
	from functions_semantic_dictionary.build_clusters import build_clusters
	
	nickname = settings_dict["nickname"]
	sg = settings_dict["sg"]
	size = settings_dict["size"]
	min_count = settings_dict["min_count"]
	hs = settings_dict["hs"]
	negative = settings_dict["negative"]
	iter = settings_dict["iter"]
	
	model_file = output_directory + nickname + ".Model.p"
	cluster_file = output_directory + nickname + ".Clusters.p"
	output_file = output_directory + nickname + ".Dictionary.txt"

	print("Starting " + nickname + " on files in " + input_directory)

	train_models(input_directory, output_file, num_workers, sg, size, min_count, hs, negative, iter)
	build_clusters(model_file, num_clusters, cluster_file)
	write_clusters(cluster_file, num_clusters, output_file)
	
	return
#--------------------------------------------------------------------------------------------#

# #Prevent pool workers from starting here#
# if __name__ == '__main__':
# #---------------------------------------#

	# import multiprocessing as mp
	# from functools import partial

	# nickname_base = "Spanish.Aranea"
	# input_directory = "../../../../data/Input/Dict_Files/Aranea.Spanish"
	# output_directory = "../../../../data/Input/Dict_Files/"
	# num_clusters = 100
	# num_workers = 4
	# num_processes = 4

	# settings_list = []
	
	# for sg in [1]:
		
		# for size in [500]:
		
			# for min_count in [500]:
			
				# for settings in [{"hs":1, "negative":0}]:
				
					# hs = settings["hs"]
					# negative = settings["negative"]
				
					# for iter in [25]:

						# nickname = nickname_base + ".SG=" + str(sg) + ".SIZE=" + str(size) + ".MIN_COUNT=" + str(min_count) + ".HS=" + str(hs) + ".ITER=" + str(iter)
						
						# settings_dict = {}
						# settings_dict["nickname"] = nickname
						# settings_dict["sg"] = sg
						# settings_dict["size"] = size
						# settings_dict["min_count"] = min_count
						# settings_dict["hs"] = hs
						# settings_dict["negative"] = negative
						# settings_dict["iter"] = iter
						
						# settings_list.append(settings_dict)
						
	# learn_dictionary(settings_list,
						# num_clusters = num_clusters,
						# num_workers = num_workers,
						# input_directory = input_directory,
						# output_directory = output_directory
						# )