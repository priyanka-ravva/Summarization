#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ## some points:

# #Reference summary & System summary 

# Here we have to check how much of System summary is captured of Reference summary.
# ---> It can be exact match overlapping
# ---> Semantically can be right [ here is the challenge how we solve this problem ]


# Basically Recall will tell how much of Reference summary is covered by System summary.

# some existing methods used Recall, Precision, and F1_score.

# [ They have base information as reference summary, with this did some experiments like N-gram overlapping, edit distance, similarity method, contextual embeddings methods ]



# But in our case,

# ##### Reference Summary == Original article(input document) & System summary == Human annotated summary

# ### Params:
# 1. Readability (Perplexity maybe?)
# 2. Creativity(Novelty) (Precision)
# 3. Relevance (Bertscore+LSI (index overlap))
# 4. Conciseness (Compression Ratio)


# ----> Relevance:
    
# how the Relevance information covered in Summary(Representation/formation): 
# how much information is copied, 
# how much information is creative,
# the whole copied & creative information is understandable or not 
# and the whole information is in concise manner or not.


# Copying     --> Precision, Recall based we can get [ Is there any other complications to identifying the copying ]

# Creativity  --> [1-Precision] - Any other problems ?? how can we improve this ??

# Conciseness --> Need to have a better telugu tokenizer

# Relevance   --> any clustering, similarity, Ranking method [ Extractive way..?? ] -- [Have to work on this....!!]



# ###########Task1: Data spliting ##################




# ###########Task2: Basic AEM pipeline #################


# Excellent Summary:

# ---> copying: , novelty: , conciseness: , Relevance:

# Good Summary:

# ---> copying: , novelty: , conciseness: , Relevance:

# Average Summary:

# ---> copying: , novelty: , conciseness: , Relevance:

# Bad Summary

# ---> copying: , novelty: , conciseness: , Relevance:













# In[2]:


from scipy import stats
import json
f = open('dataset.json',) 
corpus_info = json.load(f) 


# In[3]:


f2 = open('rouge_scores.json',) 
rouge_scores_info = json.load(f2) 


# In[4]:


with open('rouge_score_based_categorized_data_information.json') as data_file:    
    catergorized_info = json.load(data_file)


# In[5]:


################## reading the json file for further ualitative qanalysis ##############

import json

#fname = open("dataset.json",)
#corpus_dict = json.load(fname)


### loading json file
f = open('rouge_scores.json',) 
data = json.load(f) 
print("good samples =%d"%len(data))


list_samples = list(data.keys())
print(len(list_samples))
print("\n")




for i in range(0, 101, 5):
    unigrams = [ sample for sample in list_samples if(data[sample]['novelty']['rouge-1']>=i) ] 
    print("Unigrams_Novelty_range >=%d --> #of samples = %d"%(i, len(unigrams)))
print("\n---------------------------------------\n")  


for i in range(0, 101, 5):
    bigrams = [ sample for sample in list_samples if(data[sample]['novelty']['rouge-2']>=i) ] 
    print("Bigrams_Novelty_range >=%d --> #of samples = %d"%(i, len(bigrams)))
print("\n---------------------------------------\n")    


for i in range(0, 101, 5):
    trigrams = [ sample for sample in list_samples if(data[sample]['novelty']['rouge-3']>=i) ] 
    print("Trigrams_Novelty_range >=%d --> #of samples = %d"%(i, len(trigrams)))
print("\n---------------------------------------\n")    

for i in range(0, 101, 5):
    four_grams = [ sample for sample in list_samples if(data[sample]['novelty']['rouge-4']>=i) ] 
    print("4_grams_Novelty_range >=%d --> #of samples = %d"%(i, len(four_grams)))
    #print("bigrams ", len(bigrams))
print("\n---------------------------------------\n")


for i in range(0, 101, 5):
    Ngrams_L = [ sample for sample in list_samples if(data[sample]['novelty']['rouge-L']>=i) ] 
    print("Rouge_L_Novelty_range >=%d --> #of samples = %d"%(i, len(Ngrams_L)))


# In[6]:


#for i in range(0, 101, 5):
trigrams = [ sample for sample in list_samples if(data[sample]['novelty']['rouge-3']>=25) ] 
print("Trigrams_Novelty_range >=%d --> #of samples = %d"%(i, len(trigrams)))
print("\n---------------------------------------\n")    
#print(trigrams[:10])

n = 5
import random
random.seed(42)
randome_samples = random.sample(trigrams, n)

### wx formate:
from wxconv import WXC
con = WXC(order='utf2wx', lang= 'tel')



for sample_num in randome_samples:
    
    n3 = data[sample_num]['novelty']['rouge-3']
    n2 = data[sample_num]['novelty']['rouge-2']  
    n1 = data[sample_num]['novelty']['rouge-1']
    
    p1 = data[sample_num]['copy_precision']['rouge-1']
    p2 = data[sample_num]['copy_precision']['rouge-2']
    p3 = data[sample_num]['copy_precision']['rouge-3']
    
    #print(sample_num, n2, n3)
    print("Sample_num= %s, novelty_rouge1 = %.2f, novelty_rouge2 = %.2f, novelty_rouge3 = %.2f  "%(sample_num,n1,n2,n3))
    #corpus_info[ trigrams[0]['article']]
    print("article: ")
    print(corpus_info[sample_num]['article']['content'])
    print("\n")
    print("summary: ")
    print(corpus_info[ sample_num]['summary']['content'])
    print("\n---------------- WX formate ------------------\n")
    print("Article:")
    print(con.convert(corpus_info[ sample_num]['article']['content']))
    print("\n")
    print("Summary:")
    print(con.convert(corpus_info[ sample_num]['summary']['content']))
    print("\n=============================================================\n")


# In[7]:


import numpy as np
from scipy import stats

####### Novelty checking:

params = ['novelty','copy_precision','copy_recall']




for params_key in catergorized_info.keys():
    print("\n\n\n\n******************************************************** %s **********************************************************\n\n"%params_key) 
    for novelty_rouge_set in catergorized_info[params_key].keys():### keys are rouge-1,2,3,4,L:
            param_bins = catergorized_info[params_key][novelty_rouge_set] ### bins of rouge_N
            print("======================================================= %s =================================================\n\n"%novelty_rouge_set)
            for i in range(10):
                cr_thr_count = 0 ##### if we want to get for every bin:
                bin_num = (i+1)*10
                article_tokens = []
                article_sents = []

                summary_tokens = []
                summary_sents = []

                cr_sents = []
                cr_tokens = []
                
                #print("param_bins: ",param_bins[str(i)].keys())
                list_samples = param_bins[str(i)]['samples']
                #print("total samples in a Bin = %d"%len(list_samples))
                
                for sample in list_samples:
                    temp_article_tokens = corpus_info[sample]['article']['tokens_wise']
                    temp_article_sents  = corpus_info[sample]['article']['sentences_wise']

                    temp_summary_tokens = corpus_info[sample]['summary']['tokens_wise']
                    temp_summary_sents  = corpus_info[sample]['summary']['sentences_wise']

                    article_sents.append(len(temp_article_sents))
                    article_tokens.append(len(temp_article_tokens))

                    summary_sents.append(len(temp_summary_sents))
                    summary_tokens.append(len(temp_summary_tokens))

                    
                    cr_sents.append((len(temp_summary_sents)/float(len(temp_article_sents)))*100) ## sentences CR
                    cr_tokens.append((len(temp_summary_tokens)/float(len(temp_article_tokens)))*100) ## Tokens CR
                    
                    
                    
                    t1 = (len(temp_summary_tokens)/float(len(temp_article_tokens)))*100 ## tokens;
                    cr_range = 90             
                    if(t1>=cr_range):
                        #print("t1 = %d"%t1)
                        #print("sumamry tokens=%d, article tokens=%d "%((len(temp_summary_tokens), len(temp_article_tokens))))
                        cr_thr_count+=1


                ############# Tokens ################

                max_article_tokens    = max(article_tokens)
                min_article_tokens    = min(article_tokens)
                mean_article_tokens   = np.mean(np.array(article_tokens))
                median_article_tokens = np.median(article_tokens)
                mode_article_tokens   = stats.mode(article_tokens)[0][0]

                max_summary_tokens    = max(summary_tokens)
                min_summary_tokens    = min(summary_tokens)
                mean_summary_tokens   = np.mean(np.array(summary_tokens))
                median_summary_tokens = np.median(summary_tokens)
                mode_summary_tokens   = stats.mode(summary_tokens)[0][0]


                ##################### Sentences #################
                max_article_sents     = max(article_sents)
                min_article_sents     = min(article_sents)
                mean_article_sents    = np.mean(np.array(article_sents))
                median_article_sents  = np.median(article_sents)
                mode_article_sents    = stats.mode(article_sents)[0][0]
   
                
                max_summary_sents     = max(summary_sents)
                min_summary_sents     = min(summary_sents)
                mean_summary_sents    = np.mean(np.array(summary_sents))
                median_summary_sents  = np.median(summary_sents)
                mode_summary_sents    = stats.mode(summary_sents)[0][0]
                
                
                ### Compression Ratio = (length of Summary)/(length of Original Text) ######

                max_sents_cr          = max(cr_sents)
                min_sents_cr          = min(cr_sents)
                mean_sents_cr         = np.mean(cr_sents)
                median_sents_cr       = np.median(cr_sents)
                mode_sents_cr         = stats.mode(cr_sents)[0][0]
                
                max_tokens_cr         = max(cr_tokens)
                min_tokens_cr         = min(cr_tokens)
                mean_tokens_cr        = np.mean(cr_tokens)
                median_tokens_cr      = np.median(cr_tokens)
                mode_tokens_cr        = stats.mode(cr_tokens)[0][0]
                
                #print(mode_sents_cr, mode_tokens_cr)
                
                print("param = %s , bin_num = %d , total_samples_in_a_bin = %d "%(novelty_rouge_set, bin_num, len(list_samples)) )
                print("=======================================================================\n")
                print("Tokens:")
                print("max_article_tokens=%d, min_article_tokens=%d, mean_article_tokens=%d, median_article_tokens=%d, mode_article_tokens=%d "%(max_article_tokens, min_article_tokens, mean_article_tokens, median_article_tokens, mode_article_tokens)) 
                print("max_summary_tokens=%d, min_summary_tokens=%d, mean_summary_tokens=%d, median_summary_tokens=%d, mode_summary_tokens=%d "%(max_summary_tokens, min_summary_tokens, mean_summary_tokens, median_summary_tokens, mode_summary_tokens))
                print("max_tokens_cr = %.2f, min_tokens_cr = %.2f, mean_tokens_cr = %.2f, median_tokens_cr = %.2f, mode_tokens_cr = %.2f "%(max_tokens_cr, min_tokens_cr, mean_tokens_cr, median_tokens_cr, mode_tokens_cr))
                # len(list_samples)== param_bins[str(i)]['count']
                #print("total samples in a bin = %d, Tokens_CR_threshold_count = %d samples which are greater than %d CR threshold. "%(param_bins[str(i)]['count'] ,cr_thr_count, cr_range))
                print("In a bin out of %d samples, %d  samples are greater than %d tokens_CR threshold. "%(param_bins[str(i)]['count'] ,cr_thr_count, cr_range))
                print("\n")
                
                #print("Sentences:")
                #print("max_article_sents=%d, min_article_sents=%d, mean_article_sents=%d, median_article_sents=%d, mode_article_sents=%d "%(max_article_sents,min_article_sents,mean_article_sents, median_article_sents, mode_article_sents))
                #print("max_summary_sents=%d, min_summary_sents=%d, mean_summary_sents=%d, median_summary_sents=%d, mode_summary_sents=%d"%(max_summary_sents,min_summary_sents,mean_summary_sents, median_summary_sents, mode_summary_sents))
                #print("max_sents_cr=%.2f, min_sents_cr=%.2f, mean_sents_cr=%.2f, median_sents_cr=%.2f, mode_sents_cr=%.2f "%(max_sents_cr, min_sents_cr, mean_sents_cr, median_sents_cr, mode_sents_cr))

                #print("\n------------------------------------------------------------------------------------\n")

                


# In[ ]:





# In[ ]:





# In[ ]:




