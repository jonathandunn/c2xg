import statistics
import cytoolz as ct
import numpy as np
from collections import defaultdict

#-------------------------------------------------------------------#
#The main calculation function is outside of the class for jitting

def calculate_measures(lr_list, rl_list):

    #Mean Delta-P
    mean_lr = np.mean(lr_list)
    mean_rl = np.mean(rl_list)
    
    #Min Delta-P
    min_lr = np.amin(lr_list)
    min_rl = np.amin(rl_list)
    
    #Directional: Scalar and Categorical
    directional = np.subtract(lr_list, rl_list)
    directional_scalar = np.sum(directional)
    directional_categorical = min((directional > 0.0).sum(), (directional < 0.0).sum())
    
    #Beginning-Reduced Delta-P
    reduced_beginning_lr = (np.sum(lr_list) - np.sum(lr_list[1:]))
    reduced_beginning_rl = (np.sum(rl_list) - np.sum(rl_list[1:]))
    
    #End-Reduced Delta-P
    reduced_end_lr = (np.sum(lr_list) - np.sum(lr_list[0:-1]))
    reduced_end_rl = (np.sum(rl_list) - np.sum(rl_list[0:-1]))
    
    #Package and return
    return_list = [mean_lr, 
                    mean_rl, 
                    min_lr, 
                    min_rl, 
                    directional_scalar, 
                    directional_categorical, 
                    reduced_beginning_lr,
                    reduced_beginning_rl,
                    reduced_end_lr,
                    reduced_end_rl
                    ]
                    
    return return_list

#------------------------------------------------------------------#

class Association(object):

    def __init__(self, Load, nickname = "nickname"):
    
        #Initialize Ingestor
        self.Load = Load
        self.nickname = nickname
        
    #--------------------------------------------------------------#
    
    def find_discounts(self, data):
    
        #First, divide data into two equal parts
        data1 = data[:int(len(data)/2)]
        data2 = data[int(len(data)/2):]
        
        #Second, get both sets of ngrams
        ngrams1 = self.find_ngrams(data1)
        ngrams2 = self.find_ngrams(data2)
          
        discount_dict = {}
        
        #Get discounts by category
        for ngram in ngrams1:
            try:
            
                ngram_freq = ngrams1[ngram]
                ngram_type = ngram[0][0], ngram[1][0]
                
                #Get frequency of ngram in other set
                if ngram in ngrams2:
                    ngram2_freq = ngrams2[ngram]
                    
                else:
                    ngram2_freq = 0
                
                #Only three strata
                if ngram_freq > 4:
                    freq_strata = 4
                else:
                    freq_strata = ngram_freq
                        
                if ngram_type not in discount_dict:
                    discount_dict[ngram_type] = {}
                        
                if freq_strata not in discount_dict[ngram_type]:
                    discount_dict[ngram_type][freq_strata] = []
                        
                #Update discount_dict
                discount_dict[ngram_type][freq_strata].append(abs(ngram_freq - ngram2_freq))
                
            #Unigrams throw indexing errors
            except Exception as e:
                pass
            
        #Done looping through ngrams
        return_dict = {}
        for ngram_type in discount_dict:
            return_dict[ngram_type] = {}
            for freq_strata in discount_dict[ngram_type]:
                counts = discount_dict[ngram_type][freq_strata]
                avg = statistics.mean(counts)
 
                return_dict[ngram_type][freq_strata] = avg
      
        return return_dict
            
    #--------------------------------------------------------------#
    
    def process_ngrams(self, data, lex_only = False):

        #Initialize bigram dictionary
        ngrams = defaultdict(int)
        unigrams = defaultdict(int)
        total = 0

        for line in data:

            total += len(line)

            #Store unigrams
            for item in line:
                unigrams[(1, item[0])] += 1
                unigrams[(2, item[1])] += 1
                unigrams[(3, item[2])] += 1
            
            try:
                for bigram in ct.sliding_window(2, line):
                    
                    #Tuples are indexes for (LEX, SYN, SEM)
                    #Index types are 1 (LEX), 2 (SYN), 3 (SEM)
                    ngrams[((1, bigram[0][0]), (1, bigram[1][0]))] += 1    #lex_lex

                    if lex_only == False:
                        ngrams[((1, bigram[0][0]), (2, bigram[1][1]))] += 1    #lex_pos
                        ngrams[((1, bigram[0][0]), (3, bigram[1][2]))] += 1    #lex_cat
                        ngrams[((2, bigram[0][1]), (2, bigram[1][1]))] += 1    #pos_pos
                        ngrams[((2, bigram[0][1]), (1, bigram[1][0]))] += 1    #pos_lex
                        ngrams[((2, bigram[0][1]), (3, bigram[1][2]))] += 1    #pos_cat 
                        ngrams[((3, bigram[0][2]), (3, bigram[1][2]))] += 1    #cat_cat
                        ngrams[((3, bigram[0][2]), (2, bigram[1][1]))] += 1    #cat_pos
                        ngrams[((3, bigram[0][2]), (1, bigram[1][0]))] += 1    #cat_lex
            
            #Catch errors from empty lines coming out of the encoder
            except Exception as e:
                error = e

        #Note: Keep all unigrams, they are already limited by the lexicon
        
        #Reduce null indexes
        ngrams = {key: ngrams[key] for key in list(ngrams.keys()) if -1 not in key[0] and -1 not in key[1]}
        unigrams = {key: unigrams[key] for key in list(unigrams.keys()) if -1 not in key}
        
        ngrams = ct.merge([ngrams, unigrams])    
        ngrams["TOTAL"] = total
        
        del unigrams        
       
        return ngrams
    #--------------------------------------------------------------------------------------------#

    def find_ngrams(self, data, lex_only = False, n_gram_threshold = 0):

        ngrams = self.process_ngrams(data, lex_only = lex_only)
        
        #print("\tTOTAL NGRAMS: " + str(len(list(ngrams.keys()))))
        print("\tTOTAL WORDS: " + str(ngrams["TOTAL"]))
        
        #Now enforce threshold
        keepable = lambda x: x > n_gram_threshold
        ngrams = ct.valfilter(keepable, ngrams)
        
        #print("\tAfter pruning:")
        print("\tTOTAL NGRAMS: " + str(len(list(ngrams.keys()))))
            
        return ngrams

    #---------------------------------------------------------------------------------------------#

    def calculate_association(self, ngrams, normalization = False, discount_dict = False):
    
        print("\n\tCalculating association for " + str(len(list(ngrams.keys()))) + " pairs.")
        association_dict = defaultdict(dict)
        total = ngrams["TOTAL"]

        #Loop over pairs
        for key in ngrams.keys():
        
            try:
                count = ngrams.get(key, 1)
                freq_1 = ngrams.get(key[0], 1)
                freq_2 = ngrams.get(key[1], 1)
                
                #Discount count
                if normalization == True:
                    
                    #Get sequence type
                    ngram_type = key[0][0], key[1][0]
                    
                    #Get frequency strata
                    if count > 4:
                        freq_strata = 4
                    else:
                        freq_strata = count
                   
                    count = count - discount_dict[ngram_type][freq_strata]

                #a = Frequency of current pair
                a = count
                a = max(a, 0.1)
                    
                #b = Frequency of X without Y
                b = freq_1 - count
                b = max(b, 0.1)
                    
                #c = Frequency of Y without X
                c = freq_2 - count
                c = max(c, 0.1)
                    
                #d = Frequency of units without X or Y
                d = total - a - b - c
                d = max(d, 0.1)
                
                #Calculate measures    
                lr = float(a / (a + c)) - float(b / (b + d))
                rl = float(a/ (a + b)) - float(c / (c + d))    
                
                association_dict[key]["LR"] = lr
                association_dict[key]["RL"] = rl
                association_dict[key]["Freq"] = count
                
            except Exception as e:
                #print(e)
                pass
                
        print("\tProcessed " + str(len(list(association_dict.keys()))) + " items")
        
        return association_dict
    #-----------------------------------------------------------------------------------------------#
    
    def get_top(self, association_dict, direction, number):
        
        #Make initial cuts without sorting to save time
        temp_dict = {key: association_dict[key][direction] for key in association_dict.keys()}
        current_threshold = 0.25
        
        while True:
        
            above_threshold = lambda x: x > current_threshold
            temp_dict = ct.valfilter(above_threshold, temp_dict)
            
            if len(list(temp_dict.keys())) > 10000:
                current_threshold = current_threshold + 0.05
                
            else:
                break
        
        #Sort and reduce
        return_list = [(key, value) for key, value in sorted(temp_dict.items(), key=lambda x: x[1], reverse = True)]
        return_list = return_list[0:number+1]

        for key, value in return_list:
            yield key, value
        
