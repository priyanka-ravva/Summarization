#!/usr/bin/env python
# coding: utf-8

# ### Method4 [ https://pypi.org/project/rouge-metric/ ]

# In[ ]:





# # ================ Loading json files ===================

# In[1]:


#### Loading json files:

import json

f = open('rouge_scores.json',) 
samples_ngrams_scores = json.load(f) 


f2 = open("dataset.json",) 
dataset = json.load(f2) 


print("Corpus: Samples count = %d "%len(dataset.keys()))


# In[2]:




print("Corpus: Samples count = %d "%len(dataset.keys()))
print("-------------------------------------------\n")
filter_samples = [sample for sample in dataset.keys() if( len(dataset[sample]['article']['sentences_wise'])>=3   )]
print("articles sentences>=3, samples counts  = %d "%len(filter_samples))

filter_samples = [sample for sample in dataset.keys() if( len(dataset[sample]['summary']['sentences_wise'])>=3   )]
print("summaries sentences>=3, samples counts  = %d "%len(filter_samples))
print("-------------------------------------------\n")


filter_samples = [sample for sample in dataset.keys() if(((len(dataset[sample]['summary']['sentences_wise']))/float(len(dataset[sample]['article']['sentences_wise'])) )*100)<=50]
print("Sentence_CR <=50, samples counts  = %d "%len(filter_samples))
print("==============================================\n\n")

filter_samples = [sample for sample in dataset.keys() if(((len(dataset[sample]['summary']['tokens_wise']))/float(len(dataset[sample]['article']['tokens_wise'])) )*100)<=50]
print("Tokens_CR <=50, samples counts  = %d "%len(filter_samples))
print("-------------------------------------------\n")

filter_samples = [sample for sample in dataset.keys() if(((len(dataset[sample]['summary']['tokens_wise']))/float(len(dataset[sample]['article']['tokens_wise'])) )*100)<=60]
print("Tokens_CR <=60, samples counts  = %d "%len(filter_samples))
print("-------------------------------------------\n")




# In[3]:




print("samples count = %d "%len(dataset.keys()))
list_samples = list(samples_ngrams_scores.keys())

#scores: samples_ngrams_scores
#dataset: dataset


print("\nSummaries Length in terms of sentences: ")
trigrams = [ sample for sample in list_samples if(samples_ngrams_scores[sample]['novelty']['rouge-3']>=25) ] 
print("trigrams novelty greater than 25 percentages samples count = %d"%len(trigrams))

trigrams = [ sample for sample in list_samples if( (samples_ngrams_scores[sample]['novelty']['rouge-3']>=25) and(len(dataset[sample]['summary']['sentences_wise'])>=3) )] 
print("summary sentences len >=3 and trigrams novelty >=25 percentages samples count = %d"%len(trigrams))
print("\n---------------------------------------------------------\n")    


print("\nCompression Ratio(CR) in terms of Sentences:\n")
trigrams = [ sample for sample in list_samples if(samples_ngrams_scores[sample]['novelty']['rouge-3']>=25) ] 
print("trigrams novelty greater than 25 percentages samples count = %d"%len(trigrams))

trigrams = [ sample for sample in list_samples if( (samples_ngrams_scores[sample]['novelty']['rouge-3']>=25) and( (len(dataset[sample]['summary']['sentences_wise'])/float(len(dataset[sample]['article']['sentences_wise'])) )*100  )<=50 )] 
print("sentences_CR <=50 and trigrams novelty >=25 percentages samples count = %d"%len(trigrams))
print("\n---------------------------------------------------------\n")    


print("Compression Ratio(CR) in terms of Tokens:\n")
trigrams = [ sample for sample in list_samples if(samples_ngrams_scores[sample]['novelty']['rouge-3']>=25) ] 
print("trigrams novelty greater than 25 percentages samples count = %d"%len(trigrams))

trigrams = [ sample for sample in list_samples if( (samples_ngrams_scores[sample]['novelty']['rouge-3']>=25) and( (len(dataset[sample]['summary']['tokens_wise'])/float(len(dataset[sample]['article']['tokens_wise'])) )*100  )<=50 )] 
print("tokens_CR <=50 and trigrams novelty >=25 percentages samples count = %d"%len(trigrams))
#print("\n---------------------------------------\n")    

trigrams = [ sample for sample in list_samples if(samples_ngrams_scores[sample]['novelty']['rouge-3']>=25) ] 
#print("trigrams novelty greater than 25 percentages samples count = %d"%len(trigrams))

trigrams = [ sample for sample in list_samples if( (samples_ngrams_scores[sample]['novelty']['rouge-3']>=25) and( (len(dataset[sample]['summary']['tokens_wise'])/float(len(dataset[sample]['article']['tokens_wise'])) )*100  )<=60 )] 
print("tokens_CR <=60 and trigrams novelty >=25 percentages samples count = %d"%len(trigrams))
#print("\n---------------------------------------\n")    


trigrams = [ sample for sample in list_samples if(samples_ngrams_scores[sample]['novelty']['rouge-3']>=25) ] 
#print("trigrams novelty greater than 25 percentages samples count = %d"%len(trigrams))

trigrams = [ sample for sample in list_samples if( (samples_ngrams_scores[sample]['novelty']['rouge-3']>=25) and( (len(dataset[sample]['summary']['tokens_wise'])/float(len(dataset[sample]['article']['tokens_wise'])) )*100  )<=70 )] 
print("tokens_CR <=70 and trigrams novelty >=25 percentages samples count = %d"%len(trigrams))
print("\n==========================================================================================\n")    



print("\nSummaries Length in terms of sentences: ")
bigrams = [ sample for sample in list_samples if(samples_ngrams_scores[sample]['novelty']['rouge-2']>=25) ] 
print("bigrams novelty greater than 25 percentages samples count = %d"%len(bigrams))
bigrams = [ sample for sample in list_samples if( (samples_ngrams_scores[sample]['novelty']['rouge-2']>=25) and(len(dataset[sample]['summary']['sentences_wise'])>=3) )] 
print("summary sentences len >=3 and bigrams novelty >=25 percentages samples count = %d"%len(bigrams))
print("\n--------------------------------------------------------\n")    


print("\nCompression Ration(CR) in terms of Sentence:\n")
bigrams = [ sample for sample in list_samples if(samples_ngrams_scores[sample]['novelty']['rouge-2']>=25) ] 
print("bigrams novelty greater than 25 percentages samples count = %d"%len(bigrams))

bigrams =  [ sample for sample in list_samples if( (samples_ngrams_scores[sample]['novelty']['rouge-2']>=25) and( (len(dataset[sample]['summary']['sentences_wise'])/float(len(dataset[sample]['article']['sentences_wise'])) )*100  )<=50 )] 
print("sentences_CR <=50 and bigrams novelty >=25 percentages samples count = %d"%len(bigrams))
print("\n-------------------------------------------------------\n")

print("Compression Ration(CR) in terms of Tokens:\n")
bigrams = [ sample for sample in list_samples if(samples_ngrams_scores[sample]['novelty']['rouge-2']>=25) ] 
print("bigrams novelty greater than 25 percentages samples count = %d"%len(bigrams))

bigrams =  [ sample for sample in list_samples if( (samples_ngrams_scores[sample]['novelty']['rouge-2']>=25) and( (len(dataset[sample]['summary']['tokens_wise'])/float(len(dataset[sample]['article']['tokens_wise'])) )*100  )<=50 )] 
print("tokens_CR <=50 and bigrams novelty >=25 percentages samples count = %d"%len(bigrams))
#print("\n---------------------------------------\n")    
bigrams = [ sample for sample in list_samples if(samples_ngrams_scores[sample]['novelty']['rouge-2']>=25) ] 
#print("bigrams novelty greater than 25 percentages samples count = %d"%len(bigrams))

bigrams =  [ sample for sample in list_samples if( (samples_ngrams_scores[sample]['novelty']['rouge-2']>=25) and( (len(dataset[sample]['summary']['tokens_wise'])/float(len(dataset[sample]['article']['tokens_wise'])) )*100  )<=60 )] 
print("tokens_CR <=60 and bigrams novelty >=25 percentages samples count = %d"%len(bigrams))
#print("\n---------------------------------------\n")    

bigrams = [ sample for sample in list_samples if(samples_ngrams_scores[sample]['novelty']['rouge-2']>=25) ] 
#print("bigrams novelty greater than 25 percentages samples count = %d"%len(bigrams))

bigrams =  [ sample for sample in list_samples if( (samples_ngrams_scores[sample]['novelty']['rouge-2']>=25) and( (len(dataset[sample]['summary']['tokens_wise'])/float(len(dataset[sample]['article']['tokens_wise'])) )*100  )<=70 )] 
print("tokens_CR <=70 and bigrams novelty >=25 percentages samples count = %d"%len(bigrams))
print("\n---------------------------------------\n")    


# ##  Bucketing: Data Distribution 
# 
# ### Threshold : ( [25<= Novelty <=95] & [ token_CR <=60 ] ) ====> Samples_count 
# 
# ####  tokens_CR bins = [10, 20, 30, 40, 50 , 60 ]; Novelty_bins = [ 35, 45, 55, 65, 75, 85, 95 ]
# 

# In[4]:


#scores: samples_ngrams_scores
#dataset: dataset

list_samples = dataset.keys()

Novelty_bins = {i:[] for i in range(35, 106, 10)} #[ 35, 45, 55, 65, 75, 85, 95]
tokens_CR_bins = {i:[] for i in range(10, 70, 10)}#{10, 20, 30, 40, 50 , 60 }


novelty_max_limit = 100

tokens_CR_and_novelty_bins = {i:{j:[] for j in range(35, novelty_max_limit, 10)} for i in range(10, 70, 10)}#{10, 20, 30, 40, 50 , 60 }
print("intially tokens_CR_and_novelty_bins : ",tokens_CR_and_novelty_bins)

#print(tokens_CR_bins)
#print(Novelty_bins)
#print(tokens_CR_and_novelty_bins)



########################### distributing the samples ################
for sample in list_samples:
    sample_token_cr = (len(dataset[sample]['summary']['tokens_wise'])/float(len(dataset[sample]['article']['tokens_wise'])))*100
    sample_trigram_novelty_score = samples_ngrams_scores[sample]['novelty']['rouge-3']
    if(sample_token_cr<=60 and sample_trigram_novelty_score>=25 and sample_trigram_novelty_score<=95):

        #print(sample_token_cr, sample_trigram_novelty_score)

        flag1 = False
        flag2 = False
        #### checking for CR bin:
        for cr_bin in tokens_CR_bins:
            if(sample_token_cr<=cr_bin):
                tokens_CR_bins[cr_bin].append(sample)
                flag1 = True
                #### checking for novelty_bin:
                for novel_bin in Novelty_bins:
                    if(sample_trigram_novelty_score<=novel_bin):
                        Novelty_bins[novel_bin].append(sample)
                        flag2 = True
                        tokens_CR_and_novelty_bins[cr_bin][novel_bin].append(sample)
                        break
                        
                    
            if(flag1 and flag2):
                break
                    

    

    
sc = 0
print("\n\n")
for cr_bin_key in tokens_CR_and_novelty_bins.keys():
    for novelty_bin_key in range(35, novelty_max_limit, 10):
        temp=len(tokens_CR_and_novelty_bins[cr_bin_key][novelty_bin_key])
        sc+=temp
        print("cr_bin_key = %d , novelty_bin_key = %d , bin_samples_count = %d , total_samples_counts = %d"%(cr_bin_key,novelty_bin_key, temp, sc ) )
        #print(temp)
        #t = temp
        #print(t/3)
    
    
    print("--------------------------------")
    
    

    
print("\n========================================================\n")
trigrams = [ sample for sample in list_samples if(samples_ngrams_scores[sample]['novelty']['rouge-3']>=25)] 
print("Novelty >=25 percentages samples count = %d"%len(trigrams))



trigrams = [ sample for sample in list_samples if( ((len(dataset[sample]['summary']['tokens_wise'])/float(len(dataset[sample]['article']['tokens_wise'])) )*100)<=60 )] 
print("tokens_CR <=60 percentages samples count = %d"%len(trigrams))


trigrams = [ sample for sample in list_samples if( (samples_ngrams_scores[sample]['novelty']['rouge-3']>=25) and( (len(dataset[sample]['summary']['tokens_wise'])/float(len(dataset[sample]['article']['tokens_wise'])) )*100  )<=60 )] 
print("tokens_CR <=60 and trigrams novelty >=25 percentages samples count = %d"%len(trigrams))


print("\n")
print("Filitered_samples threshold is (25<= Novelty <=95) and (token_CR <=60) ")
print("corpus_size = %d,  Filitered_samples = %d "%(len(list_samples), sc))


# In[5]:


# sc = 0
# print("\n\n")
# for cr_bin_key in tokens_CR_and_novelty_bins.keys():
#     for novelty_bin_key in range(35, novelty_max_limit, 10):
#         temp=len(tokens_CR_and_novelty_bins[cr_bin_key][novelty_bin_key])
#         sc+=temp
#         print("cr_bin_key = %d , novelty_bin_key = %d , bin_samples_count = %d , total_samples_counts = %d"%(cr_bin_key,novelty_bin_key, temp, sc ) )
      
#     print("--------------------------------")
    
    


# In[6]:


import itertools
sc = 0 

test_samples = []
dev_samples = []
train_samples = []


for cr_bin_key in tokens_CR_and_novelty_bins.keys():
    for novelty_bin_key in range(35, novelty_max_limit, 10):
        temp = tokens_CR_and_novelty_bins[cr_bin_key][novelty_bin_key]
        
        data_split   = [10,10,80]# test, dev, train
        test_count   = int((data_split[0]/100)*len(temp))
        dev_count    = int((data_split[1]/100)*len(temp))
        train_count  = int((data_split[2]/100)*len(temp))
        
        #print(len(temp),test_count, dev_count, train_count)
        
        test_samples.append(temp[:test_count])
        dev_samples.append(temp[test_count: test_count+dev_count])
        train_samples.append(temp[test_count+dev_count:])
        sc+=len(temp)
        print(len(temp),test_count, dev_count, train_count, len(list(itertools.chain(*train_samples))))
        

        

test_samples = list(itertools.chain(*test_samples))
dev_samples = list(itertools.chain(*dev_samples))
train_samples = list(itertools.chain(*train_samples))

    
print(sc ,sum([len(test_samples), len(dev_samples), len(train_samples)]))        

print("\n\n")
print("test_samples = %d"%len(test_samples))
print("dev_samples = %d"%len(dev_samples))
print("train_samples = %d"%len(train_samples))

    


# In[30]:



#destination = "/home/priyanka/Desktop/Filtering/ex2_telugu_tokenizer/Equal_data_distribution/corpus/"
destination = "/home/priyanka/Desktop/Filtering/ex2_telugu_tokenizer/Equal_data_distribution/backup/corpus/"


################################## Test ###################################
te_a = open(destination+"test.article.filter_.txt","w")
te_s = open(destination+"test.title.filter_.txt","w")

print("test_samples = %d"%len(test_samples))

c = 0
for i, sample in enumerate(test_samples):
    article_content = " ".join(dataset[sample]['article']['sentences_wise']).strip()
    #article_content = article_content.lstrip()
    #article_content = " ".join(article_content)
    
    summary_content = " ".join(dataset[sample]['summary']['sentences_wise']).strip()
    #summary_content = summary_content.lstrip()
    #summary_content = " ".join(summary_content)
    
    article_content = article_content.replace("\n","")
    summary_content = summary_content.replace("\n","")
    
    te_a.write(article_content)
    te_a.write("\n")
    
    te_s.write(summary_content)
    te_s.write("\n")
    
    if(len(article_content.split("\n"))>1 or len(summary_content.split("\n"))>1 ):
        print(i, sample)
        print("article_content: ",article_content.split("\n"),len(article_content.split("\n")))
        print("\n")
        print("summary_content: ",summary_content.split("\n"), len(summary_content.split("\n")))
        print("\n********************************************\n")
    
    c+=1
    
print(c)
te_a.close()
te_s.close()


# In[9]:


import os
#destination = "/home/priyanka/Desktop/Filtering/ex2_telugu_tokenizer/Equal_data_distribution/corpus/"
destination = "/home/priyanka/Desktop/Filtering/ex2_telugu_tokenizer/Equal_data_distribution/backup/corpus/"

################################## Test ###################################
te_a = open(destination+"test.article.filter_m1.txt","w")
te_s = open(destination+"test.title.filter_m1.txt","w")

print("test_samples = %d"%len(test_samples))



c = 0
for i, sample in enumerate(test_samples):
    article_content = " ".join(dataset[sample]['article']['sentences_wise']).strip()
    #article_content = article_content.lstrip()
    #article_content = " ".join(article_content)
    
    summary_content = " ".join(dataset[sample]['summary']['sentences_wise']).strip()
    #summary_content = summary_content.lstrip()
    #summary_content = " ".join(summary_content)
    
    article_content = article_content.replace("\n","")
    summary_content = summary_content.replace("\n","")
    
    te_a.write(article_content)
    te_a.write("\n")
    
    te_s.write(summary_content)
    te_s.write("\n")
    
    sample_dir = "m1/test/"+sample ### Set wise creation:
    if not os.path.exists(sample_dir): os.makedirs(sample_dir)
    
    article_name = "article."+str(sample)+".sent.txt"#article.25276.sent.txt
    summary_name = "article."+str(sample)+".summ.sent.txt"#article.25276.summ.sent.txt

    fa = open(sample_dir+"/"+article_name,"w")
    fa.write(article_content)
    fa.close()
    
    fs = open(sample_dir+"/"+summary_name,"w")
    fs.write(summary_content)
    fs.close()
    
    if(len(article_content.split("\n"))>1 or len(summary_content.split("\n"))>1 ):
        print(i, sample)
        print("article_content: ",article_content.split("\n"),len(article_content.split("\n")))
        print("\n")
        print("summary_content: ",summary_content.split("\n"), len(summary_content.split("\n")))
        print("\n********************************************\n")
    
    c+=1
    
print(c)
te_a.close()
te_s.close()


# In[ ]:





# In[10]:



#destination = "/home/priyanka/Desktop/Filtering/ex2_telugu_tokenizer/Equal_data_distribution/corpus/"
destination = "/home/priyanka/Desktop/Filtering/ex2_telugu_tokenizer/Equal_data_distribution/backup/corpus/"


################################## Test ###################################
te_a = open(destination+"test.article.filter.txt","w")
te_s = open(destination+"test.title.filter.txt","w")

print("test_samples = %d"%len(test_samples))

c = 0
for sample in test_samples:
    article_content = dataset[sample]['article']['content'].rstrip()
    article_content = article_content.lstrip()
    article_content = article_content.strip()
    
    summary_content = dataset[sample]['summary']['content'].rstrip()
    summary_content = summary_content.lstrip()
    summary_content = summary_content.strip()
    
    article_content = article_content.replace("\n","")
    summary_content = summary_content.replace("\n","")

    sample_dir = "m1/test/"+sample ### Set wise creation:
    if not os.path.exists(sample_dir): os.makedirs(sample_dir)
    article_name = "article."+str(sample)+".sent.txt"#article.25276.sent.txt
    summary_name = "article."+str(sample)+".summ.sent.txt"#article.25276.summ.sent.txt

    fa = open(sample_dir+"/"+article_name,"w")
    fa.write(article_content)
    fa.close()
    
    fs = open(sample_dir+"/"+summary_name,"w")
    fs.write(summary_content)
    fs.close()

    
    te_a.write(str(article_content))
    te_a.write("\n")
    
    te_s.write(str(summary_content))
    te_s.write("\n")
    
    c+=1
    
print(c)
te_a.close()
te_s.close()



################################### Train ###################################
print("train_samples = %d"%len(train_samples))

tr_a = open(destination+"train.article.txt","w")
tr_s = open(destination+"train.title.txt","w")

c = 0
for sample in train_samples:
    article_content = dataset[sample]['article']['content'].rstrip()
    article_content = article_content.lstrip()
    article_content = article_content.strip()
    
    summary_content = dataset[sample]['summary']['content'].rstrip()
    summary_content = summary_content.lstrip()
    summary_content = summary_content.strip()

    
    article_content = article_content.replace("\n","")
    summary_content = summary_content.replace("\n","")

    sample_dir = "m1/train/"+sample ### Set wise creation:
    if not os.path.exists(sample_dir): os.makedirs(sample_dir)
    article_name = "article."+str(sample)+".sent.txt"#article.25276.sent.txt
    summary_name = "article."+str(sample)+".summ.sent.txt"#article.25276.summ.sent.txt

    fa = open(sample_dir+"/"+article_name,"w")
    fa.write(article_content)
    fa.close()
    
    fs = open(sample_dir+"/"+summary_name,"w")
    fs.write(summary_content)
    fs.close()

    
    tr_a.write(str(article_content))
    tr_a.write("\n")
     
    tr_s.write(str(summary_content))
    tr_s.write("\n")
    
    c+=1
    
tr_a.close()
tr_s.close()
    
print(c)

################################### Dev ##################################
print("dev_samples = %d"%len(dev_samples))

va_a = open(destination+"valid.article.filter.txt","w")
va_s = open(destination+"valid.title.filter.txt","w")

c = 0
for sample in dev_samples:
    article_content = dataset[sample]['article']['content'].rstrip()
    article_content = article_content.lstrip()
    article_content = article_content.strip()
    
    summary_content = dataset[sample]['summary']['content'].rstrip()
    summary_content = summary_content.lstrip()
    summary_content = summary_content.strip()
    

    article_content = article_content.replace("\n","")
    summary_content = summary_content.replace("\n","")

    sample_dir = "m1/dev/"+sample ### Set wise creation:
    if not os.path.exists(sample_dir): os.makedirs(sample_dir)
    article_name = "article."+str(sample)+".sent.txt"#article.25276.sent.txt
    summary_name = "article."+str(sample)+".summ.sent.txt"#article.25276.summ.sent.txt

    fa = open(sample_dir+"/"+article_name,"w")
    fa.write(article_content)
    fa.close()
    
    fs = open(sample_dir+"/"+summary_name,"w")
    fs.write(summary_content)
    fs.close()

    
    va_a.write(str(article_content))
    va_a.write("\n")
    
    va_s.write(str(summary_content))
    va_s.write("\n")
    
    c+=1
    
va_a.close()
va_s.close()
print(c)

    


# In[ ]:





# In[ ]:





# In[ ]:


########################################### completed ######################################


# In[ ]:





# In[ ]:




