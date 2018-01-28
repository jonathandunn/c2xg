# This script takes two sets of clusterings #
# and returns the similarity between them  #
# using the Measure of Concordance (MoC; Pfitzner, et al., 2009) #

import csv
import codecs
import math
import cytoolz as ct

#Set input files containing clusterings#
gold_standard_clustering_file = ''
test_clustering_file = ''

test_files = [
	'test1.txt'
	]

gold_files = [
	'test2.txt'
	]


#Count instances in each clustering file#
#-----------------------------------------#
def count_instances(input_file1,input_file2):
	
	file1_counter = 0
	file2_counter = 0
	
	fo = codecs.open(input_file1, 'rb')
	for line in fo:
		if line.strip() != '':
			file1_counter += 1
	fo.close()
	
	fo = codecs.open(input_file2, 'rb')
	for line in fo:
		if line.strip() != '':
			file2_counter += 1
	fo.close()
	
	if file1_counter != file2_counter:
		print("WARNING: Input Files Have Different Numbers of Instances")
		
	return file1_counter
#-----------------------------------------#


#Load clusterings from file into dictionary#
#-----------------------------------------#
def load_clustering(input_file):
	
	clustering = {}
	
	with open(input_file, mode='r') as infile:
		
		reader = csv.reader(infile)
		clustering = {rows[0]:rows[1] for rows in reader}

	return clustering
#-----------------------------------------#


#Count number of clusters in a given clustering#
#-----------------------------------------#
def count_clusters(clustering):

	clusters = list(set(list(clustering.values())))
	
	return len(clusters)
#-----------------------------------------#


#Calculates overlap between two given clusters#
#-----------------------------------------#
def calculate_cluster_overlap(clustering1,cluster1,clustering2,cluster2):

	cluster1 = str(cluster1)
	cluster2 = str(cluster2)

	in_cluster = lambda x: x == cluster1
	temp1 = ct.valfilter(in_cluster, clustering1)
	cluster1_inventory = list(temp1.keys())
	
	in_cluster = lambda x: x == cluster2
	temp2 = ct.valfilter(in_cluster, clustering2)
	cluster2_inventory = list(temp2.keys())
	
	shared_instances = len(list(set(cluster1_inventory) & set(cluster2_inventory)))
			
	print("Size of Cluster ", cluster1, "in Gold-Standard: ", len(cluster1_inventory))
	print("Size of Cluster ", cluster2, "in Test Set: ", len(cluster2_inventory))
	print("Number of Instances Shared Between Cluster ", cluster1, " (Gold-Standard) and Cluster ", cluster2, " (Test): ", shared_instances)
				
	if len(cluster1_inventory) != 0 and len(cluster2_inventory) !=0:
		overlap = (shared_instances * shared_instances) / (len(cluster1_inventory) * len(cluster2_inventory))
	else:
		overlap = 0
	
	print("Overlap Score for these two clusters: ", overlap, "\n")
	
	return overlap
#-----------------------------------------#


#Calculates the MoC for two given clusters#
#-----------------------------------------#
def calculate_moc(gold_clustering,gold_clusters,test_clustering,test_clusters):

	temp_overlap_list = []
	
	for gold_cluster in range(gold_clusters):
		for test_cluster in range(test_clusters):
			overlap = calculate_cluster_overlap(gold_clustering,gold_cluster,test_clustering,test_cluster)
			temp_overlap_list.append(overlap)

	raw_moc = (sum(temp_overlap_list)) - 1
	
	normalization = 1 / (math.sqrt(gold_clusters * test_clusters) - 1)
	
	moc =  normalization * raw_moc
	
	print("Normalization: ", normalization)
	print("Raw MoC: ", raw_moc)
	print("Normalized MoC: ", moc)
	
	return moc	
#-----------------------------------------#


#-----------------------------------------#
def program_flow(gold_standard_clustering_file,test_clustering_file):
	number_of_instances = count_instances(gold_standard_clustering_file,test_clustering_file)

	gold_clustering = load_clustering(gold_standard_clustering_file)
	test_clustering = load_clustering(test_clustering_file)

	gold_clusters = count_clusters(gold_clustering)
	
	test_clusters = count_clusters(test_clustering)
		
	moc = calculate_moc(gold_clustering,gold_clusters,test_clustering,test_clusters)
	
	fw = open('Results.Clustering Similarity.txt', 'a')
	fw.write(str(gold_standard_clustering_file))
	fw.write(' + ')
	fw.write(str(test_clustering_file))
	fw.write(' = ')
	fw.write(str(moc))
	fw.write('\n')
	fw.close
#------------------------------------------------------------------#

for file1 in test_files:
	for file2 in gold_files:
		program_flow(file1,file2)