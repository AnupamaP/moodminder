import numpy as np
import nltk as nk
import pandas as pd
import pickle
import re
import os

from sklearn.metrics import classification_report

import time
#http://streamhacker.com/2012/11/22/text-classification-sentiment-analysis-nltk-scikitlearn/

#1) Naive Bayes- Nice NLTK tutorial on tweet sentiment analysis
#http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/

# Next to try
#2) Maximum Entropy classifier

#3) SVM- lexicon, bigram, trigram

#DATASET
#Used- kaggle dataset- 7000 size- not generalizable
#http://help.sentiment140.com/for-students/

cur_dir = os.getcwd()

diry='stanford_data/'
diry_model=cur_dir+"/app/code/model/"
print 'diry_model:', diry_model
trainFile='train.csv'
timeit=True
label_neg=0
label_pos=4

trainSize=20000
isTrainSub=True
#-----
testFile='test.txt'
classifierName='NB_classifier'

#ADD EMOTICON!!!

def load_data(train_fileName,useC=None,isTrainSub=False):
    #http://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression
    #This regular expression strips of an URL (not just http), any punctuations, User Names or Any non alphanumeric characters.
    #It also separates the word with a single space.
##    dataFrame = pd.read_csv(train_fileName, delimiter=",",usecols=[0], names=['label','tweet'], index_col=False)

    dataFrame = pd.read_csv(train_fileName,  sep=",",usecols=useC, names=['label','tweet'], index_col=False,dtype={'label': np.int32, 'tweet': np.str_})
    tweets_list =list(dataFrame.itertuples(index=False))

    if isTrainSub and trainSize>-1:
        tweets_list=get_subsize_tweets(tweets_list)
        
##    print "\ntweets_list: ", tweets_list
    return tweets_list

def process_data(tweets_list):

    tweets=[];
    
    for(sentiment,tweet) in tweets_list:
        words_filt=process_onetweet(tweet)
        tweets.append((words_filt, sentiment))

    return tweets

def process_onetweet(tweet):
    words= re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split()
    words_filt= [e.lower() for e in words if len(e)>=3]

    return words_filt

#return tweet set with equal pos and neg
def get_subsize_tweets(tweets_list):
##    dataFrame = pd.read_csv(fileName, delimiter = "\t", names=['label','tweet'], index_col=False)
##    pos_tweets=dataFrame[(dataFrame.label==1)]
##    neg_tweets=dataFrame[(dataFrame.label==0)]
    tweets_subset=[]

    labelSet_size=trainSize/2
    pos_index=0
    neg_index=0
    
    for t in tweets_list:
        if pos_index<labelSet_size and t[0]==label_pos:
            tweets_subset.append(t)
            pos_index += 1
        elif neg_index<labelSet_size and t[0]==label_neg:
            tweets_subset.append(t)
            neg_index += 1
           
    return tweets_subset

def train_save(tweets,timeit=False):
    #remove words less than 2 characters
    #lower case
    #list of tuples

    #extract features
    wordlist= get_words_in_tweets(tweets)
    global word_feat
    word_feat= get_word_features(wordlist)

    print "extracting features..."
    training_set= nk.classify.apply_features(extract_features,tweets)

    if timeit:
        start_time = time.time()

    print "training..."
    NB_classifier= nk.NaiveBayesClassifier.train(training_set)
    
    if timeit:
        elapsed_time = time.time() - start_time
        print "elapsed_time: ",elapsed_time
        timeArr = np.array([ elapsed_time])
        np.savetxt(diry+'time_'+classifierName+'.txt', timeArr, fmt='%f')
        
    #SAVE classifier and features
    pickle.dump(word_feat, open('model/'+'feature.pickle', 'wb'))
    pickle.dump(NB_classifier, open('model/'+classifierName+'.pickle', 'wb'))

"""extract feature to create binary dictionary of whether contains words"""
def extract_features(doc,word_feat):
    doc_words= set(doc)
    features={}
    for word in word_feat:
        features['contains(%s)' % word]= (word in doc_words)
    return features

"""Extract list of distict words """
def get_words_in_tweets(tweets):

    all_words=[]

    for(words, sentiment) in tweets:
        all_words.extend(words)
        
    return all_words

"""words ranked by frequency."""
def get_word_features(wordlist):
    wordlist= nk.FreqDist(wordlist)
    word_features=wordlist.keys()
##    for word, frequency in wordlist.most_common(50):
##        print('%s;%d' % (word, frequency)).encode('utf-8')
    return word_features


def predict_save(test_tweet_list,NB_classifier,word_feat):

    test_tweets=process_data(test_tweet_list)
    
    pred_list=[]
    
    for (tweet,gold) in test_tweets:
        pred= NB_classifier.classify(extract_features(tweet,word_feat))
        pred_list.append(pred)

    gold_list=[tweet[0] for tweet in test_tweet_list]
    
    combo_result= {'gold':pd.Series(gold_list) ,\
                   'pred':pd.Series(pred_list), \
                   'tweet':pd.Series([('"'+tweet[1]+'"') for tweet in test_tweet_list]) }

    result=pd.DataFrame(combo_result)
    np.savetxt(diry+'test_result.txt', result, fmt='%i,%i,%s')

    target_names=['positive','negative']
    
    print(classification_report(gold_list, pred_list, target_names=target_names))

    return result

def predict(tweet,NB_classifier,word_feat):

    tweet=process_onetweet(tweet)

    #print "predicting..."
    pred= NB_classifier.classify(extract_features(tweet,word_feat))

    return pred

def show_results(gold, pred):

    result= nk.metrics.ConfusionMatrix(gold, pred)

def classify_set():

    fileName=diry+trainFile
##    directory='trainingandtestdata/training.1600000.processed.noemoticon.csvtrain.txt'
 #   print "loading..."
 #   tweets_list=load_data(fileName,[0,5],isTrainSub)
 #   tweets=process_data(tweets_list)
    
 #   train_save(tweets,timeit)
    #global word_feat
    word_feat= pickle.load(open('model/feature.pickle', 'rb'))
    NB_classifier= pickle.load(open('model/'+classifierName+'.pickle', 'rb'))
                                             
    ##TESTING   
    print "testing..."

    test_tweet_list=load_data(diry+testFile)
    predict_save(test_tweet_list,NB_classifier,word_feat)

    result_frame = pd.read_csv(diry+"test_result.txt", delimiter = ",", names=['gold','pred','tweet'], index_col=False)

    print "\nresult_frame: \n", result_frame
    print "\nImportant features: \n", NB_classifier.show_most_informative_features(30)    
    
    print "Finished!"

    #show_results(gold, pred)

def classify_tweet(tweet):

    fileName=diry+trainFile
    
    word_feat= pickle.load(open(diry_model+'feature.pickle', 'rb'))
    NB_classifier= pickle.load(open(diry_model+classifierName+'.pickle', 'rb'))
                                             
    ##TESTING   
    #print "classifying tweet..."
    return predict(tweet,NB_classifier,word_feat)

def test():
    tweet_list=["I like Hitler","congratulations","I hate it :(", "she is excellent.","it sucked!!"]

    for tweet in tweet_list:
        pred=classify_tweet(tweet)
        print tweet, " ",pred
        
##if __name__=='__main__':
##    main()

##    pos_tweets = [('I love this car', 'positive'),
##                  ('This view is amazing', 'positive'),
##                  ('I feel great this morning', 'positive'),
##                  ('I am so excited about the concert', 'positive'),
##                  ('He is my best friend', 'positive')]
##
##    neg_tweets = [('I do not like this car', 'negative'),
##                  ('This view is horrible', 'negative'),
##                  ('I feel tired this morning', 'negative'),
##                  ('I am not looking forward to the concert', 'negative'),
##                  ('He is my enemy', 'negative')]
##

##    def get_subsize_tweets(tweets_list):
####    dataFrame = pd.read_csv(fileName, delimiter = "\t", names=['label','tweet'], index_col=False)
####    pos_tweets=dataFrame[(dataFrame.label==1)]
####    neg_tweets=dataFrame[(dataFrame.label==0)]
##    pos_tweets=[]
##    neg_tweets=[]
##
##    labelSet_size=trainSize/2
##    pos_index=0
##    neg_index=0
##    
##    for t in tweets_list:
##        if pos_index<labelSet_size and t[0]==label_neg:
##            pos_tweets.append(t)
##            pos_index += 1
##        else if neg_index<labelSet_size and t[0]==label_pos:
##            neg_tweets.append(t)
##            neg_index += 1
##           
##    return {'pos': pos_tweets, 'neg': neg_tweets}
