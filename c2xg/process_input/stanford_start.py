#-------------------------------------------------------------------------------#
#INPUT: Memory limit, location of Stanford CoreNLP jar -------------------------#
#OUTPUT: Process Id for terminating server -------------------------------------#
#-------------------------------------------------------------------------------#
def stanford_start(Parameters):

	import subprocess
	import requests
	
	#First, see if the server is already started#
	try:
		request = requests.get('http://localhost:9000')
		print(Parameters.Stanford_Working_Directory)
	except:
		
		#First, start Stanford CoreNLP Server#
		run_string = 'javaw '
		run_string += ' -cp '
		run_string += '"*"'
		run_string += ' -Xmx'
		run_string += str(Parameters.Stanford_Memory_Limit)
		run_string += ' edu.stanford.nlp.pipeline.StanfordCoreNLPServer 9000'
	
		process_id = subprocess.Popen(run_string, cwd = Parameters.Stanford_Working_Directory, shell = False)
	
		print("Started Stanford CoreNLP annotation server.")
	
	return process_id
#--------------------------------------------------#
