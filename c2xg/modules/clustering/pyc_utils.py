"""!

@brief Utils that are used by modules of pyclustering.

@authors Andrei Novikov (pyclustering@yandex.ru)
@date 2014-2017
@copyright GNU Public License

@cond GNU_PUBLIC_LICENSE
	PyClustering is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	
	PyClustering is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
@endcond

"""

import time
import numpy
from numpy import array
from sys import platform as _platform
from numba import jit


## The number \f$pi\f$ is a mathematical constant, the ratio of a circle's circumference to its diameter.
pi = 3.1415926535


@jit(nopython = True)
def average_neighbor_distance(points, num_neigh):
	"""!
	@brief Returns average distance for establish links between specified number of nearest neighbors.
	
	@param[in] points (list): Input data, list of points where each point represented by list.
	@param[in] num_neigh (uint): Number of neighbors that should be used for distance calculation.
	
	@return (double) Average distance for establish links between 'num_neigh' in data set 'points'.
	
	"""
	
	if (num_neigh > len(points) - 1):
		raise NameError('Impossible to calculate average distance to neighbors when number of object is less than number of neighbors.')
	
	dist_matrix = [ [ 0.0 for i in range(len(points)) ] for j in range(len(points)) ]
	for i in range(0, len(points), 1):
		for j in range(i + 1, len(points), 1):
			distance = euclidean_distance(points[i], points[j])
			dist_matrix[i][j] = distance
			dist_matrix[j][i] = distance
			
		dist_matrix[i] = sorted(dist_matrix[i])

	total_distance = 0
	for i in range(0, len(points), 1):
		# start from 0 - first element is distance to itself.
		for j in range(0, num_neigh, 1):
			total_distance += dist_matrix[i][j + 1]
			
	return ( total_distance / (num_neigh * len(points)) )

@jit(nopython = True)
def centroid(points, indexes = None):
	"""!
	@brief Calculate centroid of input set of points. 
	
	@param[in] points (list): Set of points for centroid calculation.
	@param[in] indexes (list): Indexes of objects in input set of points that will be taken into account during centroid calculation.
	
	@return (list) centroid on the set of points where each element of list is corresponding to value in its dimension.
	
	"""
	
	dimension = len(points[0])
	centroid_value = [0.0] * dimension
	
	range_points = None
	if (indexes is None):
		range_points = range(len(points))
	else:
		range_points = indexes
	
	for index_point in range_points:
		centroid_value = list_math_addition(centroid_value, points[index_point])
	
	centroid_value = list_math_division_number(centroid_value, len(range_points))
	return centroid_value

@jit(nopython = True)
def median(points, indexes = None):
	"""!
	@brief Calculate geometric median of input set of points using Euclidian distance. 
	
	@param[in] points (list): Set of points for median calculation.
	@param[in] indexes (list): Indexes of objects in input set of points that will be taken into account during median calculation.
	
	@return (uint) index of point in input set that corresponds to median.
	
	"""
	
	index_median = None
	distance = float('Inf')
	
	range_points = None
	if (indexes is None):
		range_points = range(len(points))
	else:
		range_points = indexes
	
	for index_candidate in range_points:
		distance_candidate = 0.0
		for index in range_points:
			distance_candidate += euclidean_distance_sqrt(points[index_candidate], points[index])
		
		if (distance_candidate < distance):
			distance = distance_candidate
			index_median = index_candidate
	
	return index_median

@jit(nopython = True)
def euclidean_distance(a, b):
	"""!
	@brief Calculate Euclidean distance between vector a and b. 
	@details The Euclidean between vectors (points) a and b is calculated by following formula:
	
	\f[
	dist(a, b) = \sqrt{ \sum_{i=0}^{N}(b_{i} - a_{i})^{2}) }
	\f]
	
	Where N is a length of each vector.
	
	@param[in] a (list): The first vector.
	@param[in] b (list): The second vector.
	
	@return (double) Euclidian distance between two vectors.
	
	@note This function for calculation is faster then standard function in ~100 times!
	
	"""
	
	distance = euclidean_distance_sqrt(a, b)
	return distance**(0.5)

@jit(nopython = True)
def euclidean_distance_sqrt(a, b):
	"""!
	@brief Calculate square Euclidian distance between vector a and b.
	
	@param[in] a (list): The first vector.
	@param[in] b (list): The second vector.
	
	@return (double) Square Euclidian distance between two vectors.
	
	"""  
	
	#if ( ((type(a) == float) and (type(b) == float)) or ((type(a) == int) and (type(b) == int)) ):
	#	return (a - b)**2.0
	
	distance = 0.0
	for i in range(0, len(a)):
		distance += (a[i] - b[i])**2.0
		
	return distance

@jit(nopython = True)
def manhattan_distance(a, b):
	"""!
	@brief Calculate Manhattan distance between vector a and b.
	
	@param[in] a (list): The first cluster.
	@param[in] b (list): The second cluster.
	
	@return (double) Manhattan distance between two vectors.
	
	"""
	
	if ( ((type(a) == float) and (type(b) == float)) or ((type(a) == int) and (type(b) == int)) ):
		return abs(a - b)
	
	distance = 0.0
	dimension = len(a)
	
	for i in range(0, dimension):
		distance += abs(a[i] - b[i])
	
	return distance

@jit(nopython = True)
def average_inter_cluster_distance(cluster1, cluster2, data = None):
	"""!
	@brief Calculates average inter-cluster distance between two clusters.
	@details Clusters can be represented by list of coordinates (in this case data shouldn't be specified),
			 or by list of indexes of points from the data (represented by list of points), in this case 
			 data should be specified.
			 
	@param[in] cluster1 (list): The first cluster where each element can represent index from the data or object itself.
	@param[in] cluster2 (list): The second cluster where each element can represent index from the data or object itself.
	@param[in] data (list): If specified than elements of clusters will be used as indexes,
			   otherwise elements of cluster will be considered as points.
	
	@return (double) Average inter-cluster distance between two clusters.
	
	"""
	
	distance = 0.0
	
	if (data is None):
		for i in range(len(cluster1)):
			for j in range(len(cluster2)):
				distance += euclidean_distance_sqrt(cluster1[i], cluster2[j])
	else:
		for i in range(len(cluster1)):
			for j in range(len(cluster2)):
				distance += euclidean_distance_sqrt(data[ cluster1[i] ], data[ cluster2[j] ])
	
	distance /= float(len(cluster1) * len(cluster2))
	return distance ** 0.5

@jit(nopython = True)
def average_intra_cluster_distance(cluster1, cluster2, data = None):
	"""!
	@brief Calculates average intra-cluster distance between two clusters.
	@details Clusters can be represented by list of coordinates (in this case data shouldn't be specified),
			 or by list of indexes of points from the data (represented by list of points), in this case 
			 data should be specified.
	
	@param[in] cluster1 (list): The first cluster.
	@param[in] cluster2 (list): The second cluster.
	@param[in] data (list): If specified than elements of clusters will be used as indexes,
			   otherwise elements of cluster will be considered as points.
	
	@return (double) Average intra-cluster distance between two clusters.
	
	"""
		
	distance = 0.0
	
	for i in range(len(cluster1) + len(cluster2)):
		for j in range(len(cluster1) + len(cluster2)):
			first_point = None
			second_point = None
			
			if (data is None):
				# the first point
				if (i < len(cluster1)): first_point = cluster1[i]
				else: first_point = cluster2[i - len(cluster1)]
				
				# the second point
				if (j < len(cluster1)): second_point = cluster1[j]
				else: second_point = cluster2[j - len(cluster1)]
				
			else:
				# the first point
				if (i < len(cluster1)): first_point = data[ cluster1[i] ]
				else: first_point = data[ cluster2[i - len(cluster1)] ]
			
				if (j < len(cluster1)): second_point = data[ cluster1[j] ]
				else: second_point = data[ cluster2[j - len(cluster1)] ]	
			

			
			distance += euclidean_distance_sqrt(first_point, second_point)
	
	distance /= float( (len(cluster1) + len(cluster2)) * (len(cluster1) + len(cluster2) - 1.0) )
	return distance ** 0.5

@jit(nopython = True)
def variance_increase_distance(cluster1, cluster2, data = None):
	"""!
	@brief Calculates variance increase distance between two clusters.
	@details Clusters can be represented by list of coordinates (in this case data shouldn't be specified),
			 or by list of indexes of points from the data (represented by list of points), in this case 
			 data should be specified.
	
	@param[in] cluster1 (list): The first cluster.
	@param[in] cluster2 (list): The second cluster.
	@param[in] data (list): If specified than elements of clusters will be used as indexes,
			   otherwise elements of cluster will be considered as points.
	
	@return (double) Average variance increase distance between two clusters.
	
	"""
	
	# calculate local sum
	member_cluster1 = None
	member_cluster2 = None
	
	if (data is None):
		member_cluster1 = [0.0] * len(cluster1[0])
		member_cluster2 = [0.0] * len(cluster2[0])
		
	else:
		member_cluster1 = [0.0] * len(data[0])
		member_cluster2 = [0.0] * len(data[0])
	
	for i in range(len(cluster1)):
		if (data is None):
			member_cluster1 = list_math_addition(member_cluster1, cluster1[i])
		else:
			member_cluster1 = list_math_addition(member_cluster1, data[ cluster1[i] ])
	
	
	for j in range(len(cluster2)):
		if (data is None):
			member_cluster2 = list_math_addition(member_cluster2, cluster2[j])
		else:
			member_cluster2 = list_math_addition(member_cluster2, data[ cluster2[j] ])
	
	member_cluster_general = list_math_addition(member_cluster1, member_cluster2)
	member_cluster_general = list_math_division_number(member_cluster_general, len(cluster1) + len(cluster2))
	
	member_cluster1 = list_math_division_number(member_cluster1, len(cluster1))
	member_cluster2 = list_math_division_number(member_cluster2, len(cluster2))
	
	# calculate global sum
	distance_general = 0.0
	distance_cluster1 = 0.0
	distance_cluster2 = 0.0
	
	for i in range(len(cluster1)):
		if (data is None):
			distance_cluster1 += euclidean_distance_sqrt(cluster1[i], member_cluster1)
			distance_general += euclidean_distance_sqrt(cluster1[i], member_cluster_general)
			
		else:
			distance_cluster1 += euclidean_distance_sqrt(data[ cluster1[i] ], member_cluster1)
			distance_general += euclidean_distance_sqrt(data[ cluster1[i] ], member_cluster_general)
	
	for j in range(len(cluster2)):
		if (data is None):
			distance_cluster2 += euclidean_distance_sqrt(cluster2[j], member_cluster2)
			distance_general += euclidean_distance_sqrt(cluster2[j], member_cluster_general)
			
		else:
			distance_cluster2 += euclidean_distance_sqrt(data[ cluster2[j] ], member_cluster2)
			distance_general += euclidean_distance_sqrt(data[ cluster2[j] ], member_cluster_general)
	
	return distance_general - distance_cluster1 - distance_cluster2

@jit(nopython = True)
def calculate_ellipse_description(covariance, scale = 2.0):
	"""!
	@brief Calculates description of ellipse using covariance matrix.
	
	@param[in] covariance (numpy.array): Covariance matrix for which ellipse area should be calculated.
	@param[in] scale (float): Scale of the ellipse.
	
	@return (float, float, float) Return ellipse description: angle, width, height.
	
	"""
	
	eigh_values, eigh_vectors = numpy.linalg.eigh(covariance)
	order = eigh_values.argsort()[::-1]
	
	values, vectors = eigh_values[order], eigh_vectors[order]
	angle = numpy.degrees(numpy.arctan2(*vectors[:,0][::-1]))
	
	width, height = 2.0 * scale * numpy.sqrt(values)
	return angle, width, height

@jit(nopython = True)
def data_corners(data, data_filter = None):
	"""!
	@brief Finds maximum and minimum corner in each dimension of the specified data.
	
	@param[in] data (list): List of points that should be analysed.
	@param[in] data_filter (list): List of indexes of the data that should be analysed,
				if it is 'None' then whole 'data' is analysed to obtain corners.
	
	@return (list) Tuple of two points that corresponds to minimum and maximum corner (min_corner, max_corner).
	
	"""
	
	dimensions = len(data[0])
	
	bypass = data_filter
	if (bypass is None):
		bypass = range(len(data))
	
	maximum_corner = data[bypass[0]][:]
	minimum_corner = data[bypass[0]][:]
	
	for index_point in bypass:
		for index_dimension in range(dimensions):
			if (data[index_point][index_dimension] > maximum_corner[index_dimension]):
				maximum_corner[index_dimension] = data[index_point][index_dimension]
			
			if (data[index_point][index_dimension] < minimum_corner[index_dimension]):
				minimum_corner[index_dimension] = data[index_point][index_dimension]
	
	return (minimum_corner, maximum_corner)

@jit(nopython = True)
def norm_vector(vector):
	"""!
	@brief Calculates norm of an input vector that is known as a vector length.
	
	@param[in] vector (list): The input vector whose length is calculated.
	
	@return (double) vector norm known as vector length.
	
	"""
	
	length = 0.0
	for component in vector:
		length += component * component
	
	length = length ** 0.5
	
	return length

@jit(nopython = True)
def unit_vector(vector):
	"""!
	@brief Calculates unit vector.
	@details Unit vector calculates of an input vector involves two steps. 
			  The first, calculate vector length. The second,
			  divide each vector component by the obtained length.
	
	@param[in] vector (list): The input vector that is used for unit vector calculation.
	
	"""
	
	length = norm_vector(vector)
	unit_vector_instance = []
	
	for component in vector:
		unit_vector_instance.append(component / length)
	
	return unit_vector_instance

@jit(nopython = True)
def heaviside(value):
	"""!
	@brief Calculates Heaviside function that represents step function.
	@details If input value is greater than 0 then returns 1, otherwise returns 0.
	
	@param[in] value (double): Argument of Heaviside function.
	
	@return (double) Value of Heaviside function.
	
	"""
	if (value > 0.0): 
		return 1.0
	
	return 0.0

@jit(nopython = True)
def timedcall(executable_function, *args):
	"""!
	@brief Executes specified method or function with measuring of execution time.
	
	@param[in] executable_function (pointer): Pointer to function or method.
	@param[in] args (*): Arguments of called function or method.
	
	@return (tuple) Execution time and result of execution of function or method (execution_time, result_execution).
	
	"""
	
	time_start = time.clock()
	result = executable_function(*args)
	time_end = time.clock()
	
	return (time_end - time_start, result)

@jit(nopython = True)
def extract_number_oscillations(osc_dyn, index = 0, amplitude_threshold = 1.0):
	"""!
	@brief Extracts number of oscillations of specified oscillator.
	
	@param[in] osc_dyn (list): Dynamic of oscillators.
	@param[in] index (uint): Index of oscillator in dynamic.
	@param[in] amplitude_threshold (double): Amplitude threshold when oscillation is taken into account, for example,
				when oscillator amplitude is greater than threshold then oscillation is incremented.
	
	@return (uint) Number of oscillations of specified oscillator.
	
	"""
	
	number_oscillations = 0
	waiting_differential = False
	threshold_passed = False
	high_level_trigger = True if (osc_dyn[0][index] > amplitude_threshold) else False
	
	for values in osc_dyn:
		if ( (values[index] >= amplitude_threshold) and (high_level_trigger is False) ):
			high_level_trigger = True
			threshold_passed = True
		
		elif ( (values[index] < amplitude_threshold) and (high_level_trigger is True) ):
			high_level_trigger = False
			threshold_passed = True
		
		if (threshold_passed is True):
			threshold_passed = False
			if (waiting_differential is True and high_level_trigger is False):
				number_oscillations += 1
				waiting_differential = False

			else:
				waiting_differential = True
		
	return number_oscillations

@jit(nopython = True)
def allocate_sync_ensembles(dynamic, tolerance = 0.1, threshold = 1.0, ignore = None):
	"""!
	@brief Allocate clusters in line with ensembles of synchronous oscillators where each
		   synchronous ensemble corresponds to only one cluster.
	
	@param[in] dynamic (dynamic): Dynamic of each oscillator.
	@param[in] tolerance (double): Maximum error for allocation of synchronous ensemble oscillators.
	@param[in] threshold (double): Amlitude trigger when spike is taken into account.
	@param[in] ignore (bool): Set of indexes that shouldn't be taken into account.
	
	@return (list) Grours (lists) of indexes of synchronous oscillators, for example, 
			[ [index_osc1, index_osc3], [index_osc2], [index_osc4, index_osc5] ].
			
	"""
	
	descriptors = [] * len(dynamic)
	
	# Check from the end for obtaining result
	for index_dyn in range(0, len(dynamic[0]), 1):
		if ((ignore is not None) and (index_dyn in ignore)):
			continue
		
		time_stop_simulation = len(dynamic) - 1
		active_state = False
		
		if (dynamic[time_stop_simulation][index_dyn] > threshold):
			active_state = True
			
		# if active state is detected, it means we don't have whole oscillatory period for the considered oscillator, should be skipped.
		if (active_state is True):
			while ( (dynamic[time_stop_simulation][index_dyn] > threshold) and (time_stop_simulation > 0) ):
				time_stop_simulation -= 1
			
			# if there are no any oscillation than let's consider it like noise
			if (time_stop_simulation == 0):
				continue
			
			# reset
			active_state = False
		
		desc = [0, 0, 0] # end, start, average time of oscillation
		for t in range(time_stop_simulation, 0, -1):
			if ( (dynamic[t][index_dyn] > 0) and (active_state is False) ):
				desc[0] = t
				active_state = True
			elif ( (dynamic[t][index_dyn] < 0) and (active_state is True) ):
				desc[1] = t
				active_state = False
				
				break
		
		if (desc == [0, 0, 0]):
			continue
		
		desc[2] = desc[1] + (desc[0] - desc[1]) / 2.0
		descriptors.append(desc)
		
	
	# Cluster allocation
	sync_ensembles = []
	desc_sync_ensembles = []
	
	for index_desc in range(0, len(descriptors), 1):
		if (descriptors[index_desc] == []):
			continue
		
		if (len(sync_ensembles) == 0):
			desc_ensemble = descriptors[index_desc]
			reducer = (desc_ensemble[0] - desc_ensemble[1]) * tolerance
			
			desc_ensemble[0] = desc_ensemble[2] + reducer
			desc_ensemble[1] = desc_ensemble[2] - reducer
			
			desc_sync_ensembles.append(desc_ensemble)
			sync_ensembles.append([ index_desc ])
		else:
			oscillator_captured = False
			for index_ensemble in range(0, len(sync_ensembles), 1):
				if ( (desc_sync_ensembles[index_ensemble][0] > descriptors[index_desc][2]) and (desc_sync_ensembles[index_ensemble][1] < descriptors[index_desc][2])):
					sync_ensembles[index_ensemble].append(index_desc)
					oscillator_captured = True
					break
				
			if (oscillator_captured is False):
				desc_ensemble = descriptors[index_desc]
				reducer = (desc_ensemble[0] - desc_ensemble[1]) * tolerance
		
				desc_ensemble[0] = desc_ensemble[2] + reducer
				desc_ensemble[1] = desc_ensemble[2] - reducer
		
				desc_sync_ensembles.append(desc_ensemble)
				sync_ensembles.append([ index_desc ])
	
	return sync_ensembles
	
@jit(nopython = True)  
def linear_sum(list_vector):
	"""!
	@brief Calculates linear sum of vector that is represented by list, each element can be represented by list - multidimensional elements.
	
	@param[in] list_vector (list): Input vector.
	
	@return (list|double) Linear sum of vector that can be represented by list in case of multidimensional elements.
	
	"""
	dimension = 1
	linear_sum = 0.0
	list_representation = (type(list_vector[0]) == list)
	
	if (list_representation is True):
		dimension = len(list_vector[0])
		linear_sum = [0] * dimension
		
	for index_element in range(0, len(list_vector)):
		if (list_representation is True):
			for index_dimension in range(0, dimension):
				linear_sum[index_dimension] += list_vector[index_element][index_dimension]
		else:
			linear_sum += list_vector[index_element]

	return linear_sum
	
@jit(nopython = True)
def square_sum(list_vector):
	"""!
	@brief Calculates square sum of vector that is represented by list, each element can be represented by list - multidimensional elements.
	
	@param[in] list_vector (list): Input vector.
	
	@return (double) Square sum of vector.
	
	"""
	
	square_sum = 0.0
	list_representation = (type(list_vector[0]) == list)
		
	for index_element in range(0, len(list_vector)):
		if (list_representation is True):
			square_sum += sum(list_math_multiplication(list_vector[index_element], list_vector[index_element]))
		else:
			square_sum += list_vector[index_element] * list_vector[index_element]
		 
	return square_sum

@jit(nopython = True)  
def list_math_subtraction(a, b):
	"""!
	@brief Calculates subtraction of two lists.
	@details Each element from list 'a' is subtracted by element from list 'b' accordingly.
	
	@param[in] a (list): List of elements that supports mathematical subtraction.
	@param[in] b (list): List of elements that supports mathematical subtraction.
	
	@return (list) Results of subtraction of two lists.
	
	"""
	return [a[i] - b[i] for i in range(len(a))]

@jit(nopython = True)
def list_math_substraction_number(a, b):
	"""!
	@brief Calculates subtraction between list and number.
	@details Each element from list 'a' is subtracted by number 'b'.
	
	@param[in] a (list): List of elements that supports mathematical subtraction.
	@param[in] b (list): Value that supports mathematical subtraction.
	
	@return (list) Results of subtraction between list and number.
	
	"""		
	return [a[i] - b for i in range(len(a))]  

@jit(nopython = True)
def list_math_addition(a, b):
	"""!
	@brief Addition of two lists.
	@details Each element from list 'a' is added to element from list 'b' accordingly.
	
	@param[in] a (list): List of elements that supports mathematic addition..
	@param[in] b (list): List of elements that supports mathematic addition..
	
	@return (list) Results of addtion of two lists.
	
	"""	
	return [a[i] + b[i] for i in range(len(a))]

@jit(nopython = True)
def list_math_addition_number(a, b):
	"""!
	@brief Addition between list and number.
	@details Each element from list 'a' is added to number 'b'.
	
	@param[in] a (list): List of elements that supports mathematic addition.
	@param[in] b (double): Value that supports mathematic addition.
	
	@return (list) Result of addtion of two lists.
	
	"""	
	return [a[i] + b for i in range(len(a))]

@jit(nopython = True)
def list_math_division_number(a, b):
	"""!
	@brief Division between list and number.
	@details Each element from list 'a' is divided by number 'b'.
	
	@param[in] a (list): List of elements that supports mathematic division.
	@param[in] b (double): Value that supports mathematic division.
	
	@return (list) Result of division between list and number.
	
	"""	
	return [a[i] / b for i in range(len(a))]

@jit(nopython = True)
def list_math_division(a, b):
	"""!
	@brief Division of two lists.
	@details Each element from list 'a' is divided by element from list 'b' accordingly.
	
	@param[in] a (list): List of elements that supports mathematic division.
	@param[in] b (list): List of elements that supports mathematic division.
	
	@return (list) Result of division of two lists.
	
	"""	
	return [a[i] / b[i] for i in range(len(a))]

@jit(nopython = True)
def list_math_multiplication_number(a, b):
	"""!
	@brief Multiplication between list and number.
	@details Each element from list 'a' is multiplied by number 'b'.
	
	@param[in] a (list): List of elements that supports mathematic division.
	@param[in] b (double): Number that supports mathematic division.
	
	@return (list) Result of division between list and number.
	
	"""	
	return [a[i] * b for i in range(len(a))]

@jit(nopython = True)
def list_math_multiplication(a, b):
	"""!
	@brief Multiplication of two lists.
	@details Each element from list 'a' is multiplied by element from list 'b' accordingly.
	
	@param[in] a (list): List of elements that supports mathematic multiplication.
	@param[in] b (list): List of elements that supports mathematic multiplication.
	
	@return (list) Result of multiplication of elements in two lists.
	
	"""		
	return [a[i] * b[i] for i in range(len(a))]
