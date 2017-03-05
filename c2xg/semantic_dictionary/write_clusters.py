#----------------------------------------------------------------------------------#
#Write clusters in readable comma separated format --------------------------------#
#----------------------------------------------------------------------------------#
def write_clusters(input_clusters, 
					num_clusters, 
					output_file
					):

	import pickle
	import os
	import codecs

	#Open clusters#
	with open(input_clusters, 'rb') as f:
		word_centroid_map = pickle.load(f)
	
	print("Done loading clusters")

	fw = codecs.open(output_file, "w", encoding = "utf-8")

	for i in range(num_clusters):

		for key in word_centroid_map.keys():
	
			if word_centroid_map[key] == i:
		
				fw.write(str(key))
				fw.write(",")
				fw.write(str(word_centroid_map[key]))
				fw.write("\n")

	fw.close()
	
	return
#--------------------------------------------------------------------------------#