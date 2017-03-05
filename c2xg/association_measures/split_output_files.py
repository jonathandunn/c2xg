#-----------------------------------------------------------#
#--- Split output files into sub-sets for each process -----#
#-----------------------------------------------------------#
def split_output_files(seq, num):
	
	if len(seq) < num:
		num = len(seq) / 2
	
	avg = len(seq) / float(num)
	out = []
	last = 0.0

	while last < len(seq):
		out.append(seq[int(last):int(last + avg)])
		last += avg
		
	if len(out[-1]) == 1:
		del out[-1]

	return out
#------------------------------------------------------------#