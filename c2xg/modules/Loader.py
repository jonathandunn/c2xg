import os
import pickle
import codecs
from random import randint

#The loader object handles all file access to enable local or S3 bucket support
class Loader(object):

	def __init__(self, input, output, s3 = False, s3_bucket = ""):
	
		#if using S3, input and output dirs are prefixes
		self.input_dir = input
		self.output_dir = output
		self.s3 = s3
		self.s3_bucket = s3_bucket
			
	#---------------------------------------------------------------#
	
	def save_file(self, file, filename):
	
		if self.s3 == True:
		
			#Initialize boto3 client
			import boto3
			client = boto3.client("s3")
		
			#Write file to disk
			temp_name = "temp." + str(randint(1,10000000)) + ".p"	#Have to avoid conflicts across cores
			with open(os.path.join(self.output_dir, temp_name), "wb") as handle:
				pickle.dump(file, handle, protocol = pickle.HIGHEST_PROTOCOL)
				
			#Upload and delete
			client.upload_file(temp_name, self.s3_bucket, filename)
			os.remove(temp_name)
		
		else:
		
			#Write file to disk
			with open(os.path.join(self.output_dir, filename), "wb") as handle:
				pickle.dump(file, handle, protocol = pickle.HIGHEST_PROTOCOL)
				
	#---------------------------------------------------------------#
	
	def list_input(self):
	
		files = []	#Initiate list of files
		
		#If listing an S3 bucket
		if self.s3 == True:
		
			#Initialize boto3 client
			import boto3
			client = boto3.client("s3")
				
			#Find all files in directory
			response = client.list_objects_v2(
							Bucket = self.s3_bucket,
							Delimiter = "/",
							Prefix = self.input_dir + "/"
							)
		
			for key in response["Contents"]:
				files.append(key["Key"])
			
		#If reading local file	
		else:
		
			for filename in os.listdir(self.input_dir):
				files.append(filename)
				
		return [x for x in files if x.endswith(".txt")]
			
	#---------------------------------------------------------------#
	
	def list_output(self, type = ""):
	
		files = []	#Initiate list of files
		
		#If listing an S3 bucket
		if self.s3 == True:
		
			#Initialize boto3 client
			import boto3
			client = boto3.client("s3")
				
			#Find all files in directory
			response = client.list_objects_v2(
							Bucket = self.s3_bucket,
							Delimiter = "/",
							Prefix = self.output_dir + "/"
							)
		
			for key in response["Contents"]:
				if type in key:
					files.append(key["Key"])
			
		#If reading local file	
		else:
		
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
	
		if self.s3 == True:
		
			#Initialize boto3 client
			import boto3
			client = boto3.client("s3")
				
			#Find all files in directory
			response = client.list_objects_v2(
							Bucket = self.s3_bucket,
							Delimiter = "/",
							Prefix = self.output_dir + "/"
							)
		
			files = []
			for key in response["Contents"]:
				files.append(key["Key"])
			
			#Check for file specified
			if filename in files:	
			
				#Download, load and return
				temp_name = "temp." + str(randint(1,10000000)) + ".p"	#Have to avoid conflicts across cores
				client.download_file(self.s3_bucket, filename, temp_name)
				
				with open(temp_name, "rb") as handle:
					return_file = pickle.load(handle)
					
				os.remove(temp_name)
				
				return return_file
			
			#If file isn't found in the S3 bucket, return error
			else:
				print(filename + " not found")

		#If reading local file	
		else:
		
			with open(os.path.join(self.output_dir, filename), "rb") as handle:
					return_file = pickle.load(handle)
				
			return return_file
			
	#---------------------------------------------------------------#
	
	def read_file(self, file):
	
		#Read from S3 bucket
		if self.s3 == True:
		
			#Initialize boto3 client
			import boto3
			client = boto3.client("s3")
			
			temp_name = "temp." + str(randint(1,10000000)) + ".txt"	#Have to avoid conflicts across cores
			client.download_file(self.s3_bucket, file, temp_name)
				
			with codecs.open(temp_name, "rb") as fo:
				lines = fo.readlines()
					
			os.remove(temp_name)
				
			for line in lines:
				line = line.decode("utf-8")
				yield line
				
		#Read local directory
		else:
		
			with codecs.open(os.path.join(self.input_dir, file), "rb") as fo:
				lines = fo.readlines()
					
			for line in lines:
				line = line.decode("utf-8", errors = "replace")
				yield line
	
	#---------------------------------------------------------------#
	
	def clean(self, type = ""):
	
		files_to_remove = []
		
		for filename in os.listdir(self.output_dir):
		
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
			os.remove(os.path.join(self.output_dir, file))		