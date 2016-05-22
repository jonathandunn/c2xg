#-------------------------------------------------------------------------------#
#INPUT: Process ID of StanfordCoreNLP server -----------------------------------#
#OUTPUT: None ------------------------------------------------------------------#
#-------------------------------------------------------------------------------#
def stanford_stop(process_id):

	import subprocess

	subprocess.Popen.terminate(process_id)
	print("Stopped Stanford CoreNLP annotation server.")
	
	return
#--------------------------------------------------#