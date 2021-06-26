import os
import pickle
import codecs
import gzip
import time
from random import randint

#The loader object handles all file access to enable local or S3 bucket support
class Loader(object):

	def __init__(self, input, output, language, max_words = False):
	
		#if using S3, input and output dirs are prefixes
		if input != None:
			self.input_dir = os.path.join(input, language)
			self.output_dir = os.path.join(output, language)

		else:
			self.input_dir = None
			self.output_dir = None

		self.language = language
		self.max_words = max_words
		
		#Check that directories exist
		if input != None:
			
			if os.path.isdir(self.input_dir) == False:
				os.makedirs(self.input_dir)
				print("Creating input folder")
			
			if os.path.isdir(self.output_dir) == False:
				os.makedirs(self.output_dir)
				print("Creating output folder")
			
	#---------------------------------------------------------------#
	
	def save_file(self, file, filename):
		
		print("\t\tSaving " + filename)

		#Write file to disk
		try:
			with open(os.path.join(self.output_dir, filename), "wb") as handle:
				pickle.dump(file, handle, protocol = 3)
					
		except:
			time.sleep(100)
			with open(os.path.join(self.output_dir, filename), "wb") as handle:
				pickle.dump(file, handle, protocol = 3)
				
	#---------------------------------------------------------------#
	
	def list_input(self):
	
		files = []	#Initiate list of files
		
		for filename in os.listdir(self.input_dir):
			files.append(filename)
				
		return [x for x in files if x.endswith(".txt") or x.endswith(".gz")]
			
	#---------------------------------------------------------------#
	
	def list_output(self, type = ""):
	
		files = []	#Initiate list of files
		
		for filename in os.listdir(self.output_dir):
			if type in filename:
				files.append(filename)
				
		return files
			
	#---------------------------------------------------------------#
	
	def check_file(self, filename):
	
		file_list = self.list_output()
		
		if filename in file_list:
			return True
			
		else:
			return False
	#--------------------------------------------------------------#
	
	def load_file(self, filename):
	
		try:
			with open(os.path.join(self.output_dir, filename), "rb") as handle:
				return_file = pickle.load(handle)
		except Exception as e:
			print(filename, e)
				
			with open(os.path.join(self.output_dir, filename), "rb") as handle:
				return_file = pickle.load(handle)
				
		return return_file
	
	#---------------------------------------------------------------#
	
	def read_file(self, file):
	
		max_counter = 0

		if file.endswith(".txt"):

			with codecs.open(os.path.join(self.input_dir, file), "rb") as fo:
				lines = fo.readlines()

		elif file.endswith(".gz"):
				
			with gzip.open(os.path.join(self.input_dir, file), "rb") as fo:
				lines = fo.readlines()

		for line in lines:
			line = line.decode("utf-8", errors = "replace")

			if self.max_words != False:
				if max_counter < self.max_words:
					max_counter += len(line.split())
					yield line
							
			else:
				yield line
				
	#---------------------------------------------------------------#
	
	def clean(self, type = ""):
	
		print("\nNow cleaning up after learning cycle.")
		files_to_remove = []
		
		#First, cleaning method if using local data
		for filename in os.listdir(self.output_dir):
			filename = self.output_dir + "/" + filename
			if type == "ngrams" or type == "":
				if "ngrams" in filename:
					files_to_remove.append(filename)
				
			elif type == "association" or type == "":
				if "association" in filename:
					files_to_remove.append(filename)
				
			elif type == "candidates" or type == "":
				if "candidates" in filename:
					files_to_remove.append(filename)
						
		for file in files_to_remove:
			if "Final_Grammar" not in file:
				print("\t\tRemoving " + file)
				os.remove(os.path.join(self.output_dir, file))

	#---------------------------------------------------------------#
			