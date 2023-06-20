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

    def __init__(self, in_dir = None, out_dir = None,
                    nickname = "", max_words = False, 
                    phrases = False, sg_model = False, cbow_model = False,
                    max_sentence_length = 50):
    
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
        self.clips = None
        self.starting_index = 0
        self.max_sentence_length = max_sentence_length

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
    
    def read_file(self, file, iterating = False):
    
        max_counter = 0
        starting_counter = 0
        clean_lines = []

        #Read lines from uncompressed text file
        if file.endswith(".txt"):
            with codecs.open(os.path.join(self.in_dir, file), "rb") as fo:
                lines = fo.readlines()
        #Read lines from compressed text file
        elif file.endswith(".gz"):
            with gzip.open(os.path.join(self.in_dir, file), "rb") as fo:
                lines = fo.readlines()
        
        #Get data up to max words
        if iterating == False:
            for line in lines:
                #Ensure utf-8 input
                line = line.decode("utf-8", errors = "replace")
                starting_counter += len(line.split())
                
                if starting_counter > self.starting_index:
                    #Control the amount of input data
                    if self.max_words != False:
                        if max_counter < self.max_words:
                            if len(line) > 2:
                                max_counter += len(line.split())
                                clean_lines.append(line)
                            
        #Get data up to max words
        elif iterating != False:
            #Define number of words to discard first and then the stopping point
            start = iterating[0]
            stop = iterating[1]

            #Iterate over lines
            for line in lines:
                #Ensure utf-8 input
                line = line.decode("utf-8", errors = "replace")
                starting_counter += len(line.split())
                
                if starting_counter > self.starting_index:
                    if len(line) > 2:
                        max_counter += len(line.split())
                    
                        #Start after existing data is passed
                        #In between, accumulate data
                        if max_counter > start and max_counter < stop:
                            clean_lines.append(line)
                        
        return clean_lines
                
    #---------------------------------------------------------------#
    def get_unk(self, word, type = "cbow"):

        word = str(word)
        
        if type == "cbow":
            vector = self.cbow_model[word]
            centroids = [self.cbow_centroids[x] for x in sorted(self.cbow_centroids.keys())]
            distances = pairwise_distances(vector.reshape(1, -1), centroids, metric="cosine", n_jobs=1) 
            index = np.argmin(distances)
            self.cbow_encode[word] = index
            
        elif type == "sg":
            vector = self.sg_model[word]
            centroids = [self.sg_centroids[x] for x in sorted(self.sg_centroids.keys())]
            distances = pairwise_distances(vector.reshape(1, -1), centroids, metric="cosine", n_jobs=1)
            index = np.argmin(distances)
            self.sg_encode[word] = index
        
        return index
    
    #---------------------------------------------------------------#
    
    def decode(self, item):

        #1 = LEX; 2 = SYN; 3 = SEM
        try:
            rep = item[0]
            index = item[1]
            word = self.lex_decode.get(index, "UNK")

            #Get lexical item
            if rep == 1:
                value = word
                
            #Get cbow category
            elif rep == 2:
                #Get existing category
                try:
                    value = self.cbow_decode[index]

                #Or use embedding to assign to category
                except Exception as e:
                    value = self.get_unk(word, "cbow")
                    value = self.cbow_decode[value]
                    print("152", e)
                    
                value = "syn:" + str(index) + "_" + str(value)
            
            #Get sg category
            elif rep == 3:
                #Get existing category
                try:
                    value = self.sg_decode[index]
                #Or use embedding to assign category
                except Exception as e:
                    value = self.get_unk(word, "sg")
                    value = self.sg_decode[value]
                    print("165", e)
                    
                value = "sem:" + str(index) + "_" + str(value)
          
        #Catch items that are improperly formatted
        except Exception as e:
            value = "UNK"
            
        return value        

    #---------------------------------------------------------------------------#
    
    def decode_construction(self, construction, clips = None):

        #Input may be a string rather than tuple
        if isinstance(construction, str):
            construction = eval(construction)
            
        #No specific clips are passed, used default
        if clips == None:
            clips = self.clips

        #Initialize empty string
        construction_string = "[ "
        clip_index = False
        
        #Check for clipping info
        if clips != None:
            if construction in clips:
                clip_index = True
            else:
                clip_index = False

        #Iterate over slots
        for i in range(len(construction)):
            current_slot = construction[i]

            #Lex
            if current_slot[0] == 1:
                if current_slot[1] in self.lex_decode:
                    value = self.lex_decode[current_slot[1]]
                    current_slot = "lex:" + str(value)
                else:
                    current_slot = "lex:" + "missing"
                
            #Syn
            elif current_slot[0] == 2:
                if current_slot[1] in self.cbow_decode:
                    value = self.cbow_decode[current_slot[1]]
                    current_slot = "syn:" + str(current_slot[1]) + "_" + str(value)
                else:
                    current_slot = "syn:" + "missing"
                
            #Sem
            elif current_slot[0] == 3:
                if current_slot[1] in self.sg_decode:
                    value = self.sg_decode[current_slot[1]]
                    current_slot = "sem:" + str(current_slot[1]) + "_" + str(value)
                else:
                    current_slot = "sem:" + "missing"
                
            #Add constraint to string
            construction_string += current_slot
            
            #Get transition symbol
            if i+1 < len(construction):
            
                #Add clip notation if necessary
                if clip_index == True and i in clips[construction]:
                    transition = " ] " + clips[construction][i] + " [ "
                
                #Get transition symbol if no clipping
                else:
                
                    pair = (construction[i], construction[i+1])
                    
                    #For grammars in previous corpora, transition may be missing
                    try:
                        assoc = self.assoc_dict[pair[0]][pair[1]]
                        difference = assoc["LR"] - assoc["RL"]
                        
                        #LR is stronger
                        if difference > 0.1:
                            transition = " > "
                        #RL is stronger
                        elif difference < -0.1:
                            transition = " < "
                        #Neither dominates
                        else:
                            transition = " -- "
                    
                    #By default, no direction
                    except:
                        transition = " -- "
                        
                #Add transition to construction
                construction_string += transition
          
        construction_string += " ]"
        
        return construction_string
        
    #---------------------------------------------------------------------------#
    
    def load(self, input_file, mode = "files"):

        #If only got one file, wrap in list
        #if isinstance(input_file, str):
        #    input_file = [input_file]
        
        if mode == "files":
            lines = self.read_file(input_file)
            lines = [self.clean(x) for x in lines]
            
        elif mode == "lines":
            lines = [self.clean(x) for x in input_file]
            
        return lines
            
    #---------------------------------------------------------------------------#
    
    #Create categories dictionaries are annotating new corpora
    def add_categories(self, cbow_df, sg_df, lexicon, phrases, full_lexicon, unique_words, update = False):
    
        #Convert unique words from df to lsit
        unique_words = unique_words.loc[:,"Word"].tolist()
        
        #Load dictionaries if possible
        cbow_filename = self.nickname + ".cbow_dict.p"
        sg_filename = self.nickname + ".sg_dict.p"
        cbow_encode_filename = self.nickname + ".cbow_encode.p"
        cbow_decode_filename = self.nickname + ".cbow_decode.p"
        sg_encode_filename = self.nickname + ".sg_encode.p"
        sg_decode_filename = self.nickname + ".sg_decode.p"
        lex_encode_filename = self.nickname + ".lex_encode.p"
        lex_decode_filename = self.nickname + ".lex_decode.p"
        
        #Check if exist
        if not os.path.exists(os.path.join(self.out_dir, cbow_filename)):
            
            print("Creating encoder/decoder resources.")
    
            #Dictionaries with ranking info
            self.cbow = {}
            self.sg = {}
            
            #Encoding and decoding dictionaries
            self.cbow_encode = {}
            self.cbow_decode = {}
            self.sg_encode = {}
            self.sg_decode = {}
            self.lex_encode = {}
            self.lex_decode = {}
            
            #Create dictionary for the local (cbow) category for each word
            for row in cbow_df.itertuples():
                #Get row
                index = row[0]
                rank = row[1]
                word = row[2]
                category = row[3]
                category_name = row[4]
                #Add to dictionary
                self.cbow[word] = {}
                self.cbow[word]["Category"] = category
                self.cbow[word]["Similarity"] = rank
                self.cbow[word]["Index"] = index
                self.lex_encode[word] = index
                self.lex_decode[index] = word
                self.cbow_encode[word] = category
                self.cbow_decode[category] = category_name
            
            #Create dictionary for the non-local (sg) category for each word
            for row in sg_df.itertuples():
                #Get row
                index = row[0]
                rank = row[1]
                word = row[2]
                category = row[3]
                category_name = row[4]
                self.sg_decode[category] = category_name
                #Add to dictionary, but not unique words
                if word not in unique_words:
                    self.sg[word] = {}
                    self.sg[word]["Category"] = category
                    self.sg[word]["Similarity"] = rank
                    self.sg_encode[word] = category
                
            #Get centroids for each cbow category for OOV words
            if self.cbow_centroids == False:
                print("Creating centroids for local categories")
                cbow_centroids = {}
                for category, category_df in cbow_df.groupby("Category"):
                    category_name = str(category_df.loc[:,"Category_Name"].tolist()[0])
                    if "unique" not in category_name:
                        words = category_df.loc[:,"Category"].tolist()
                        ranks = category_df.loc[:,"Rank"].tolist()
                        current_centroid =  self.cbow_model.get_mean_vector(keys=words, weights=ranks, pre_normalize=True, post_normalize=False)
                        cbow_centroids[category] = current_centroid
                #Centroids as a list where the index = the cluster id
                self.cbow_centroids = {}
                for i in range(len(cbow_centroids)):
                    if i in cbow_centroids:
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
                        current_centroid =  self.sg_model.get_mean_vector(keys=words, weights=ranks, pre_normalize=True, post_normalize=False)
                        sg_centroids[category] = current_centroid
                #Centroids as a list where the index = the cluster id
                self.sg_centroids = {}
                for i in range(len(sg_centroids)):
                    if i in sg_centroids:
                        self.sg_centroids[i] = sg_centroids[i]
                    
            #Assign OOV words to classes
            for word in full_lexicon.loc[:,"Word"].tolist():

                if word not in self.cbow_encode:
                    temp_cbow = self.get_unk(word, type = "cbow")
                    self.cbow_encode[word] = temp_cbow
                    
                    #Don't get semantic domains for unique words
                    if word not in unique_words:
                        temp_sg = self.get_unk(word, type = "sg")
                        self.sg_encode[word] = temp_sg
            
            #Save to file
            self.save_file(self.cbow, cbow_filename)
            self.save_file(self.sg, sg_filename)
            self.save_file(self.cbow_encode, cbow_encode_filename)
            self.save_file(self.cbow_decode, cbow_decode_filename)
            self.save_file(self.sg_encode, sg_encode_filename)
            self.save_file(self.sg_decode, sg_decode_filename)
            self.save_file(self.lex_encode, lex_encode_filename)
            self.save_file(self.lex_decode, lex_decode_filename)
    
        #Load from file
        else:
            print("Loading encoding/decoding resources")
            self.cbow = self.load_file(cbow_filename)
            self.sg = self.load_file(sg_filename)
            self.cbow_encode = self.load_file(cbow_encode_filename)
            self.cbow_decode = self.load_file(cbow_decode_filename)
            self.sg_encode = self.load_file(sg_encode_filename)
            self.sg_decode = self.load_file(sg_decode_filename)
            self.lex_encode = self.load_file(lex_encode_filename)
            self.lex_decode = self.load_file(lex_decode_filename)
            
            #Update with new lexical items
            if update == True:
            
                #Get starting size
                starting_sizes = (len(self.lex_encode), len(self.cbow_encode), len(self.sg_encode))
                
                #Get next lex integer
                max_index = max(list(self.lex_encode.values())) + 1
            
                #Process new words
                for word in full_lexicon.loc[:,"Word"].tolist():
                    if word not in self.lex_encode:
                    
                        self.lex_encode[word] = max_index
                        self.lex_decode[max_index] = word
                        max_index += 1
                
                        if word not in self.cbow_encode:
                            temp_cbow = self.get_unk(word, type = "cbow")
                            self.cbow_encode[word] = temp_cbow
                            
                            #Don't get semantic domains for unique words
                            if word not in unique_words:
                                temp_sg = self.get_unk(word, type = "sg")
                                self.sg_encode[word] = temp_sg
                                
                #Display results and save
                ending_sizes = (len(self.lex_encode), len(self.cbow_encode), len(self.sg_encode))
                print("\tExpanding lexicon-cbow-sg from " + str(starting_sizes) + " to " + str(ending_sizes))
                
                #Save to file
                self.save_file(self.cbow, cbow_filename)
                self.save_file(self.sg, sg_filename)
                self.save_file(self.cbow_encode, cbow_encode_filename)
                self.save_file(self.cbow_decode, cbow_decode_filename)
                self.save_file(self.sg_encode, sg_encode_filename)
                self.save_file(self.sg_decode, sg_decode_filename)
                self.save_file(self.lex_encode, lex_encode_filename)
                self.save_file(self.lex_decode, lex_decode_filename)

    #---------------------------------------------------------------------------#
        
    def clean(self, line, encode=True):

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

        line = line.replace("nbsp"," ").strip()
        line = line.split()
        
        if len(line) > self.max_sentence_length:
            line = line[:self.max_sentence_length]
            

        #If phrases have been learned, find them
        if self.phrases != False:
            line = self.phrases[line]
            
        #If categories have been learned, add them
        if encode == True:
            if self.cbow != False and self.sg != False:
                line = [self.enrich(x) for x in line]

        return line

    #---------------------------------------------------------------------------#
    def enrich(self, word):
    
        #Lex
        try:
            index = self.lex_encode[word]

        except:
            index = -1
        
        #Syn
        try:
            syn = self.cbow_encode[word]

        #OOV, get category by embedding
        except:
            syn = self.get_unk(word, type = "cbow")
            
        #Check sg dictionary
        try:
            sem = self.sg_encode[word]

        #OOV, get category
        except:
            sem = self.get_unk(word, type = "sg")
            
        #Return line as tuple (LEX, SYN, SEM)
        return (index, syn, sem)
    
    #---------------------------------------------------------------------------#

    def get_lexicon(self, input_data, npmi_threshold = 0.75, min_count = 1, max_vocab = None):

        if isinstance(input_data, str):
            
            #Get data
            data = self.read_file(input_data)
            data = [self.clean(line, encode = False) for line in data]

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
            full_lexicon_df.loc[:,"Actual_Frequency"] = full_lexicon_df.loc[:,"Frequency"]
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
                
            #Reduce to max vocab
            if max_vocab != None:
                lexicon = dict(sorted(lexicon.items(), key=lambda x: x[1], reverse=True)[:max_vocab])
                print("REDUCED ", len(lexicon))
            
            return lexicon, phrases, unique_words, full_lexicon_df

    #---------------------------------------------------------------------------#
            