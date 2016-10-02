#-----------------------------------------------------------#
#--- Split output files into sub-sets for each process -----#
#-----------------------------------------------------------#
def split_output_files(seq, num):
	
	avg = len(seq) / float(num)
	out = []
	last = 0.0

	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg

	return out
#------------------------------------------------------------#