import os
import pickle
import codecs
import gzip
import time
import cytoolz as ct
from cleantext import clean
from gensim.models.phrases import Phrases

#The loader object handles all file access
class Loader(object):

    def __init__(self, in_dir = None, out_dir = None, language = "eng", max_words = False, phrases = None):
    
        self.language = language
        self.max_words = max_words
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.phrases = phrases
        
        #Check that directories exist
        if in_dir != None:
            
            if os.path.isdir(self.in_dir) == False:
                os.makedirs(self.in_dir)
                print("Creating input folder")
            
        if out_dir != None:
            if os.path.isdir(self.out_dir) == False:
                os.makedirs(self.out_dir)
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
    
        files = []    #Initiate list of files
        
        for filename in os.listdir(self.input_dir):
            files.append(filename)
                
        return [x for x in files if x.endswith(".txt") or x.endswith(".gz")]
            
    #---------------------------------------------------------------#
    
    def list_output(self, type = ""):
    
        files = []    #Initiate list of files
        
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

            with codecs.open(os.path.join(self.in_dir, file), "rb") as fo:
                lines = fo.readlines()

        elif file.endswith(".gz"):
                
            with gzip.open(os.path.join(self.in_dir, file), "rb") as fo:
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
    
    def clean_files(self, filetype = ""):
    
        print("\nNow cleaning up after learning cycle.")
        files_to_remove = []
        
        #First, cleaning method if using local data
        for filename in os.listdir(self.out_dir):
            filename = self.out_dir + "/" + filename
            if filetype == "ngrams" or filetype == "":
                if "ngrams" in filename:
                    files_to_remove.append(filename)
                
            elif filetype == "association" or filetype == "":
                if "association" in filename:
                    files_to_remove.append(filename)
                
            elif filetype == "candidates" or filetype == "":
                if "candidates" in filename:
                    files_to_remove.append(filename)
                        
        for file in files_to_remove:
            if "Final_Grammar" not in file:
                print("\t\tRemoving " + file)
                os.remove(os.path.join(self.out_dir, file))

    #---------------------------------------------------------------#
        
    def build_decoder(self):

        #Create a decoding resource
        #LEX = 1, POS = 2, CAT = 3
        decoding_dict = {}
        decoding_dict[1] = self.word_dict
        decoding_dict[2] = self.pos_dict
        decoding_dict[3] = {key: "<" + str(key) + ">" for key in list(set(self.domain_dict.values()))}
            
        self.decoding_dict = decoding_dict

    #---------------------------------------------------------------------------#
    
    def decode(self, item):
    
        sequence = [self.decoding_dict.get([pair[0]][pair[1]], "UNK") for pair in item]
            
        return " ".join(sequence)        

    #---------------------------------------------------------------------------#
    
    def decode_construction(self, item):
        
        sequence = []
        for pair in item:

            try:
                val = self.decoding_dict[pair[0]][pair[1]]
            except:
                val = "UNK"

            sequence.append(val)
            
        return "[ " + " -- ".join(sequence) + " ]"

    #---------------------------------------------------------------------------#
    
    def load(self, input_files, no_file = False):

        if no_file == False:

            #If only got one file, wrap in list
            if isinstance(input_files, str):
                input_files = [input_files]
        
            for file in input_files:
                for line in self.read_file(file):
                    if len(line) > 1:
                        line = self.clean(line)
                        yield line

        elif no_file == True:
            for line in input_files:
                    if len(line) > 1:
                        line = self.clean(line)
                        yield line
                      
    #---------------------------------------------------------------------------#
        
    def clean(self, line):

        #Use clean-text
        line = clean(line,
                        fix_unicode = True,
                        to_ascii = False,
                        lower = True,
                        no_line_breaks = True,
                        no_urls = True,
                        no_emails = True,
                        no_phone_numbers = True,
                        no_numbers = True,
                        no_digits = True,
                        no_currency_symbols = True,
                        no_punct = True,
                        replace_with_punct = "",
                        replace_with_url = "<URL>",
                        replace_with_email = "<EMAIL>",
                        replace_with_phone_number = "<PHONE>",
                        replace_with_number = "<NUMBER>",
                        replace_with_digit = "0",
                        replace_with_currency_symbol = "<CUR>"
                        )

        line = line.split()

        if self.phrases != None:
            line = self.phrases[line]

        return line

    #---------------------------------------------------------------------------#

    def get_lexicon(self, input_data, npmi_threshold = 0.75, min_count = 1):

        if isinstance(input_data, str):
            
            data = [x for x in self.load(input_data)]

            #Find phrases, then freeze
            phrase_model = Phrases(data, min_count = min_count, threshold = npmi_threshold, scoring = "npmi", delimiter = " ")
            phrases = phrase_model.freeze()
            
            #Reduce lexicon to min_count
            threshold = lambda x: x >= min_count
            lexicon = ct.valfilter(threshold, phrase_model.vocab)
            
            #Remove unkept phrases from lexicon
            remove_list = []
            for key in lexicon:
                if " " in key and key not in phrases.phrasegrams:
                    remove_list.append(key)
            for key in remove_list:
                lexicon.pop(key)

            self.phrases = phrases

            return lexicon, phrases

    #---------------------------------------------------------------------------#
            