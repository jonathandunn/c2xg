import os
import pickle
import codecs
import gzip
import time
import cytoolz as ct
import pandas as pd
import numpy as np
import multiprocessing as mp
from cleantext import clean
from gensim.models.phrases import Phrases
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances

#The loader object handles all file access
class Loader(object):

    def __init__(self, in_dir = None, out_dir = None, workers = 1,
                    nickname = "", language = "eng", max_words = False, 
                    phrases = False, sg_model = False, cbow_model = False):
    
        self.language = language
        self.max_words = max_words
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.phrases = phrases
        self.nickname = nickname
        self.cbow = False
        self.sg = False
        self.sg_model = sg_model
        self.cbow_model = cbow_model
        self.cbow_centroids = False
        self.sg_centroids = False
        self.workers = workers

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
            with open(os.path.join(self.out_dir, filename), "wb") as handle:
                pickle.dump(file, handle, protocol = 3)
                    
        except:
            time.sleep(100)
            with open(os.path.join(self.out_dir, filename), "wb") as handle:
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
    
    def load_file(self, filename):
    
        try:
            with open(os.path.join(self.out_dir, filename), "rb") as handle:
                return_file = pickle.load(handle)
        except Exception as e:
            print(filename, e)
                
            try:
                with open(os.path.join(self.out_dir, filename), "rb") as handle:
                    return_file = pickle.load(handle)
            except:
                return_file = False
                
        return return_file
    
    #---------------------------------------------------------------#
    
    def read_file(self, file):
    
        max_counter = 0
        clean_lines = []

        #Read lines from uncompressed text file
        if file.endswith(".txt"):
            with codecs.open(os.path.join(self.in_dir, file), "rb") as fo:
                lines = fo.readlines()
        #Read lines from compressed text file
        elif file.endswith(".gz"):
            with gzip.open(os.path.join(self.in_dir, file), "rb") as fo:
                lines = fo.readlines()
        
        #Ensure utf-8 input
        for line in lines:
            line = line.decode("utf-8", errors = "replace")
            #Control the amount of input data
            if self.max_words != False:
                if max_counter < self.max_words:
                    if len(line) > 2:
                        max_counter += len(line.split())
                        clean_lines.append(line)
                            
        return clean_lines
                
    #---------------------------------------------------------------#
    def get_unk(self, word, type = "cbow"):
    
        if type == "cbow":
            vector = self.cbow_model.wv[word]
            centroids = [self.cbow_centroids[x] for x in sorted(self.cbow_centroids.keys())]
            distances = pairwise_distances(vector.reshape(1, -1), centroids, metric="cosine", n_jobs=1) 
            
        elif type == "sg":
            vector = self.sg_model.wv[word]
            centroids = [self.sg_centroids[x] for x in sorted(self.sg_centroids.keys())]
            distances = pairwise_distances(vector.reshape(1, -1), centroids, metric="cosine", n_jobs=1)
        
        return np.argmin(distances)
    
    #---------------------------------------------------------------#
    
    def decode(self, item):
        
        #1 = LEX; 2 = SYN; 3 = SEM
        try:
            rep = item[0]
            index = item[1]
            word = self.indexes.get(index, "UNK")
            
            #Get lexical item
            if rep == 1:
                value = word
                
            #Get cbow category
            elif rep == 2:
                #Get existing category
                try:
                    value = self.cbow[word]["Category"]
                #Or use embedding to assign to category
                except:
                    value = self.get_unk(word, "cbow")
                    
                value = "syn: " + self.cbow_names[value]
            
            #Get sg category
            elif rep == 3:
                #Get existing category
                try:
                    value = self.sg[word]["Category"]
                #Or use embedding to assign category
                except:
                    value = self.get_unk(word, "sg")
                    
                value = "sem: " + self.sg_names[value]
          
        #Catch items that are improperly formatted
        except Exception as e:
            value = "UNK"
            
        return value        

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
    
    def get_decoder(self):
    
        #Get reverse dictionaries for decoding
        self.cbow_dict = {val: key for (key, val) in self.cbow_names.items()}
        self.sg_dict = {val: key for (key, val) in self.sg_names.items()}
        self.word_dict = {val: key for (key, val) in self.indexes.items()}
        
    #---------------------------------------------------------------------------#
    
    def load(self, input_file):

        #If only got one file, wrap in list
        if isinstance(input_file, str):
            lines = self.read_file(input_file)
            lines = [self.clean(x) for x in lines]
            
            return lines
            
    #---------------------------------------------------------------------------#
    
    #Create categories dictionaries are annotating new corpora
    def add_categories(self, cbow_df, sg_df):
    
        self.cbow = {}
        self.sg = {}
        self.cbow_names = {}
        self.sg_names = {}
        self.indexes = {}
        
        #Create dictionary for the local (cbow) category for each word
        for row in cbow_df.itertuples():
            index = row[0]
            rank = row[1]
            word = row[2]
            category = row[3]
            category_name = row[4]
            self.cbow[word] = {}
            self.cbow[word]["Category"] = category
            self.cbow[word]["Similarity"] = rank
            self.cbow[word]["Index"] = index
            self.indexes[index] = word
            self.cbow_names[category] = category_name
        
        #Create dictionary for the non-local (sg) category for each word
        for row in sg_df.itertuples():
            index = row[0]
            rank = row[1]
            word = row[2]
            category = row[3]
            category_name = row[4]
            self.sg[word] = {}
            self.sg[word]["Category"] = category
            self.sg[word]["Similarity"] = rank
            self.sg[word]["Index"] = index
            self.sg_names[category] = category_name
            
        #Get centroids for each cbow category for OOV words
        if self.cbow_centroids == False:
            print("Creating centroids for local categories")
            cbow_centroids = {}
            for category, category_df in cbow_df.groupby("Category"):
                category_name = str(category_df.loc[:,"Category_Name"].tolist()[0])
                if "unique" not in category_name:
                    words = category_df.loc[:,"Category"].tolist()
                    ranks = category_df.loc[:,"Rank"].tolist()
                    current_centroid =  self.cbow_model.wv.get_mean_vector(keys=words, weights=ranks, pre_normalize=True, post_normalize=False)
                    cbow_centroids[category] = current_centroid
            #Centroids as a list where the index = the cluster id
            self.cbow_centroids = {}
            for i in range(len(cbow_centroids)):
                self.cbow_centroids[i] = cbow_centroids[i]
                
        #Get centroids for each sg category for OOV words
        if self.sg_centroids == False:
            print("Creating centroids for non-local categories")
            sg_centroids = {}
            for category, category_df in sg_df.groupby("Category"):
                category_name = str(category_df.loc[:,"Category_Name"].tolist()[0])
                if "unique" not in category_name:
                    words = category_df.loc[:,"Category"].tolist()
                    ranks = category_df.loc[:,"Rank"].tolist()
                    current_centroid =  self.sg_model.wv.get_mean_vector(keys=words, weights=ranks, pre_normalize=True, post_normalize=False)
                    sg_centroids[category] = current_centroid
            #Centroids as a list where the index = the cluster id
            self.sg_centroids = {}
            for i in range(len(sg_centroids)):
                self.sg_centroids[i] = sg_centroids[i]
    
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

        #If phrases have been learned, find them
        if self.phrases != False:
            line = self.phrases[line]
            
        #If categories have been learned, add them
        if self.cbow != False and self.sg != False:
            line = [self.enrich(x) for x in line]

        return line

    #---------------------------------------------------------------------------#
    def enrich(self, word):
    
        #If the word is in the cbow dictionary, get its index
        try:
            syn = self.cbow[word]["Category"]
            index = self.cbow[word]["Index"]
        #OOV, get category by embedding
        except:
            syn = self.get_unk(word, type = "cbow")
            index = -1
            
        #Check sg dictionary
        try:
            sem = self.sg[word]["Category"]
        #OOV, get category
        except:
            sem = self.get_unk(word, type = "sg")
            
        #Return line as tuple (LEX, SYN, SEM)
        return (index, syn, sem)
    
    #---------------------------------------------------------------------------#

    def get_lexicon(self, input_data, npmi_threshold = 0.75, min_count = 1):

        if isinstance(input_data, str):
            
            data = self.load(input_data)

            #Find phrases, then freeze
            phrase_model = Phrases(data, min_count = min_count, threshold = npmi_threshold, scoring = "npmi", delimiter = " ")
            phrases = phrase_model.freeze()

            #Get full lexicon without potential phrases
            full_lexicon = phrase_model.vocab
            threshold = lambda x: " " not in x
            full_lexicon = ct.keyfilter(threshold, full_lexicon)

            #Get Zipfian statistics about the full lexicon
            total_size = sum(full_lexicon.values())
            full_lexicon_df = pd.DataFrame.from_dict(full_lexicon, orient="index")
            full_lexicon_df.reset_index(inplace=True)
            full_lexicon_df.columns = ["Word", "Frequency"]
            full_lexicon_df.sort_values("Frequency", inplace=True, ascending=False)
            full_lexicon_df.loc[:,"Rank"] = full_lexicon_df.loc[:,"Frequency"].rank(axis=0, method='average', ascending=False)
            full_lexicon_df.to_csv(os.path.join(self.out_dir, self.nickname+".full_lexicon.csv"), index=False)

            #Get a list of words which account for over 1% of the total data
            threshold = total_size/100
            unique_words = full_lexicon_df[full_lexicon_df.loc[:,"Frequency"] > threshold]

            #Prepare to estimate Zipfian distribution
            full_lexicon_df.loc[:,"Frequency"] = np.log10(full_lexicon_df.loc[:,"Frequency"])
            full_lexicon_df.loc[:,"Rank"] = np.log10(full_lexicon_df.loc[:,"Rank"])

            #Do regression and plot against actual data
            regr = smf.ols(formula = 'Frequency ~ Rank', data = full_lexicon_df)
            res = regr.fit()

            pred_ols = res.get_prediction()
            iv_l = pred_ols.summary_frame()["obs_ci_lower"]
            iv_u = pred_ols.summary_frame()["obs_ci_upper"]
            fig, ax = plt.subplots(figsize=(8, 6))
            y = full_lexicon_df.loc[:,"Frequency"]
            x = full_lexicon_df.loc[:,"Rank"]
            ax.plot(x, y, "o", label="Data")
            ax.plot(x, res.fittedvalues, "r--.", alpha=0.25, label="Model")
            ax.plot(x, iv_u, "r--", alpha = 0.25)
            ax.plot(x, iv_l, "r--", alpha = 0.25)
            ax.legend(loc="best")
            plt.xlabel("Rank (Log)")
            plt.ylabel("Frequency (Log)")
            plt.savefig(os.path.join(self.out_dir, self.nickname+".lexicon_distribution.png"), dpi=300, bbox_inches = "tight")
            
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

            #Save phrases for future cleaning
            self.phrases = phrases

            return lexicon, phrases, unique_words

    #---------------------------------------------------------------------------#
            