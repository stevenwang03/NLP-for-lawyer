# Load library
import pandas as pd
import numpy as np
import time
import os
import sys
import pickle
import re
import os.path
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models import doc2vec
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stops = set(stopwords.words("french"))


""""Read an extracted pharse and remove word that are not in the vocabulary model 
	Args:
    sentence: the extracted pharse;
    model: the train model;
  Returns:
    ooV: the list of all out-of-vocabulary words.
"""
def verifWordNotIntModel(sentence, model):
    ooV=[]
    vocabulary = list(model.wv.vocab)
    for word in sentence:
        if(word not in vocabulary):
            ooV.append(word)
    return ooV

""""Compute the similarity between all extracted pharse and all given topics with word move distance 
	Args:
    sentence: the extracted pharse;
    model: the train model;
  Returns:
    ooV: the list of all out-of-vocabulary words.
"""
def computeScore(alljudgements,listOfTopics,model):
    scores = np.zeros((len(alljudgements), len(listOfTopics)))   

    i=0
    for text in alljudgements:
        list_ooV = verifWordNotIntModel(text, model)
        for w in list_ooV:
            text.remove(w)
            
        j=0
        for topic in listOfTopics:
            if(len(text)==0):
                score = 0
            else:
                score = model.n_similarity(text, topic)
            scores[i][j]=score
            j += 1
        i += 1
    return scores

def computeScoreWmd(alljudgements,listOfTopics,model):
	nb_judgments = len(alljudgements)
	scores = np.zeros((nb_judgments, len(listOfTopics))) 
	instance = WmdSimilarity(alljudgements, model)   

	i=0
	sec_old = time.time()
	for text in alljudgements:
		list_ooV = verifWordNotIntModel(text, model)
		for w in list_ooV:
			text.remove(w)
            
		j=0
		for topic in listOfTopics:
			distance = model.wmdistance(text, topic)
			scores[i][j]=distance
			j += 1
		i += 1
		if i%1000==0:
			sec_new = time.time()
			print(sec_new-sec_old," s elapsed. ",i+1,"/",nb_judgments)
			sec_old = sec_new
	return scores	

def normalize_dist(dist,cent_nb1,cent_nb2):
	alpha = []
	beta = []
	nb_col = dist.shape[1]
	dist_norm = np.copy(dist)
	for i in range(nb_col):
		centile1 = np.percentile(dist[:,i],100-cent_nb1)
		centile2 = np.percentile(dist[:,i],100-cent_nb2)
		c1 = (100.-cent_nb1)/cent_nb1
		c2 = (100.-cent_nb2)/cent_nb2
		beta.append(np.log(c1/c2)/np.log(centile1/centile2))
		alpha.append(c1/(centile1**beta[i]))
		dist_norm[:,i] = alpha[i]*dist[:,i]**beta[i]
	return dist_norm,alpha,beta
	
def normalize_scores(scores):
    score_norm = 1./(1.+scores)
    max_col = []
    for j in range(score_norm.shape[1]):
        max_col.append(score_norm[:,j].max())
        score_norm[:,j]/=max_col[j]
        
    pickle.dump(max_col,open('max_scores.p','wb'))
    return score_norm
	
def loadRefMaxScores():
    max_col=pickle.load(open("max_scores.p","rb"))
    return max_col
	

def tableResult(list_index,list_extractedpharses,scores):
    dict_final={'line':[],'text':[]}

    for i in range(0,len(list_index)):
        dict_final['line'].append(list_index[i])
        dict_final['text'].append(list_extractedpharses[i])
    
    dict_final['sim_a']=scores[:,0]
    dict_final['sim_b']=scores[:,1]
    dict_final['sim_c']=scores[:,2]
    dict_final['sim_d']=scores[:,3]
    dict_final['sim_e']=scores[:,4]
    dict_final['sim_f']=scores[:,5]
    dict_final['sim_g']=scores[:,6]

    df = pd.DataFrame.from_dict(dict_final)
    return df
	
	
def sortTable(df):
    #this function might take some time
    nb_phrase = df.values.shape[0]
    list_gap = np.zeros((nb_phrase,))-1
    a = list(set(df.values[:,1])) # Extracting non repeated phrases
    nb_unique_phrases = len(a)
    count=1
    for phrase in a:
        if count%1000 ==0:
            print(count,'/',nb_unique_phrases)
        i = np.where(df.values[:,1]==phrase)[0]
        ind_sort = np.argsort(df.values[i[0],2:])
        list_gap[i] = df.values[i[0],ind_sort[-1]+2]-df.values[i[0],ind_sort[-2]+2]
        count+=1
    ind = np.where(list_gap==-1)
    ind_sort_gap = np.argsort(list_gap)
    df = df.reindex(ind_sort_gap)
    pickle.dump(df,open('tableSort.p','wb'))
    return df
 
 
def trainingModel(alljudgements, listOfTopics, sentencesForTraining):
    model = Doc2Vec(dm = 1, min_count=1, window=10, vector_size=150, sample=1e-4, negative=10)

    #use all the extracted phrases of all files 
    phrases=[]
    for line in alljudgements:
        phrases.append(line)
    for line in listOfTopics:
        phrases.append(line)
    for line in sentencesForTraining:
        phrases.append(line)

    sentences = [doc2vec.TaggedDocument(sentence, 'tag') for sentence in phrases]
    model.build_vocab(sentences)

    for epoch in range(500):
        model.train(sentences,epochs=model.epochs,total_examples=model.corpus_count)
        seconds = time.time()
        print("Seconds since epoch =", seconds)
        print("Epoch # {} is complete.".format(epoch+1))
        if(epoch%30==0):
            #save model
            model.save('doc2vec2.model')
			

def ReadTrainingDecisions(repertoryName):
    
    listOfSentences=[]

    for dirpath, dirnames, filenames in os.walk(repertoryName):
        for filename in [f for f in filenames if f.endswith(".xml")]:
            text=open(os.path.join(dirpath, filename),'r', encoding='utf-8').read()
            contenu0=text.split('<CONTENU>')
            contenu1=contenu0[1].split('</CONTENU>')
            sentences=contenu1[0].split('<br/>')
            listOfSentences+=sentences

    pickle.dump(listOfSentences,open('listeSentences.p','wb'))
	
	
def loadTrainingDecisions():
    listOfSentences=pickle.load(open("listeSentences.p","rb"))
    return listOfSentences
	

def preprocessingInputs(df):
	count = 0
	alljudgements=[]
	list_index=[]
	list_extractedpharses=[]
	
	
	for judgement in df['text']:
		#remove numbers
		extract = re.sub(r'[0-9]+', '', judgement)
		#Remove ponctuation
		cleanText = re.sub(r'[^\w\s]',' ',extract)
		allLinesOfText = cleanText.lower().split('\n\n')
		viewPharses=judgement.split('\n\n')

		#words add after some verification of data
		stops.update(["leurs", "ainsi", "toutes", "xxxx","après","aa"])

		for line, sentence in zip(allLinesOfText, viewPharses):
			tokens = word_tokenize(line, language = 'french')
			#Remove Stopwords
			stopped_tokens=[w for w in tokens if not w in stops]
			if(len(stopped_tokens)>0):
				alljudgements.append(stopped_tokens)
				list_index.append(count)
				list_extractedpharses.append(sentence)
		count += 1        
	return alljudgements, list_index, list_extractedpharses	
	

def formatTrainingDecisions(listOfSentences):
    
    sentencesForTraining=[]

    for sentence in listOfSentences:
        #remove numbers
        sentence = re.sub(r'[0-9]+', '', sentence)
        #Remove ponctuation
        sentence = re.sub(r'[^\w\s]',' ',sentence)
        tokens = word_tokenize(sentence, language = 'french')
        #Remove Stopwords
        stopped_tokens=[w for w in tokens if not w in stops]
        if(len(stopped_tokens)>0):
            sentencesForTraining.append(stopped_tokens)
    return sentencesForTraining

