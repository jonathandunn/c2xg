import numpy as np
from modules.pyclustering.pyc_xmeans import xmeans
from modules.pyclustering.pyc_center_initializer import kmeans_plusplus_initializer

data = np.random.rand(2000,20)
initial_centers = kmeans_plusplus_initializer(data, 5).initialize()
xmeans_instance = xmeans(data, initial_centers, kmax = 5000, ccore = False)
xmeans_instance.process()
clusters = xmeans_instance.get_clusters()

for x in clusters:
	print(x)