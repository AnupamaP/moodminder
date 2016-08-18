#!/usr/local/bin/python

"""
TWO tasks
1. Read tweets in as a stream/past and classify as pos or neg
2. Recommend activity using POS tagging in twitter-sim-score.py
"""

import sys, random, json, tweepy
import StdOutListener
from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
import TweetRecommend as tr
#from Emotion import classify_tweet
import TweetClassify
from emoji import Emoji
import re 

class TweetProcess(object): 
	def __init__(self, t_handle, classifier, view):
		#Variables that contains the user credentials to access Twitter API 
		access_token = "716775869046513664-5A4VLHy6O2AlxgPYnnDqK68hUgovsea"
		access_token_secret = "9imA5sFl6koFoweWeDbMzti0EwQs3b2gb5JszsAic4DSB"
		consumer_key = "XTb6wIyh9M2ZXaeNgIQcZ52JX"
		consumer_secret = "6XLYcm4YT2aAqxEpd9Ym4AZaW12rBRNsoXHNRwvLW3g1hG0L5B"
		
		# Authenticate app
		self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
		self.auth.set_access_token(access_token, access_token_secret)

		# Get twitter handle and id
		self.twitter_handle = t_handle
		self.api = tweepy.API(self.auth)
		user = self.api.get_user(screen_name = self.twitter_handle)
		self.twitter_id = user.id

		# Initialize database
		self.DB = {}
		self.DB['food'] = ['candy']
		self.DB['music'] = ['coldplay']
		self.DB['joke'] = ['knock knock.. who is']

		# Initialize classifier
		self.classifier = classifier.classify_tweet #(function)

		# Set view
		self.view = view # (function in view.py)
	
	def tweet_process(self, stream, request):
		if int(stream):
			# Streaming input
			l = StdOutListener.StdOutListener(self.twitter_handle, self.classifier, self.view)
			stream = tweepy.Stream(self.auth, l)
			stream.filter(follow=[str(self.twitter_id)])
		else:
			tweets = self.api.user_timeline(screen_name = self.twitter_handle, count = 1, include_rts = True)
			#tweets = [	'I hate this cream biscuit',
    	   	#		  	'I hate this song',
       	   	#		  	'This strawberry is bad',
       	   	#   		'Jazz is the best worst kind of music',
       	   	#  		  	'I like this day',	
       	   	#   		'I am happy',
        	#  			'The man is fat',
        	#  			'I like cake',
        	#  			'I like bread',
        	#  			'Listening to music while eating food',
        	# 			 'I like racism']
			
			for t in tweets:
				# changed to return -- anupama
				t.text = (t.text).encode('utf8').decode('utf8')
				classification = u'\U0001f61f'
				if self.classifier(t.text):
					classification = u'\U0001f604'
					self.view({'name': self.twitter_handle, 'tweets': t.text, 
					   'sentiment': classification, 'recommendation':'Be happy'})
					#return {'name': self.twitter_handle, 'tweets': t.text, 
					#   'sentiment': classification, 'recommendation':'Be happy'}
				else:	
					similarity_score, category = tr.get_category(tr.tweet_filter(t.text), self.DB.keys())
					if similarity_score < 0.2: # similarity_score [0,1]
						category = 'joke'
					self.view({'name': self.twitter_handle, 'tweets': t.text, 
						   'sentiment': classification, 'recommendation': (random.choice(self.DB[category])).upper()})
					#return {'name': self.twitter_handle, 'tweets': t.text, 
					#	   'sentiment': classification, 'recommendation': (random.choice(self.DB[category])).upper()}
				#print t.text+' '+str(self.classifier(t.text))
				

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print 'USAGE:', sys.argv[0], '<twitter_handle> <1=stream 0=past>'
		sys.exit(0)

	def view(out):
		print 'View-TweetProcess: ', out
		
	twitter_handle = sys.argv[1]
	stream = int(sys.argv[2])

	tp = TweetProcess(twitter_handle)
	tp.tweet_process(stream)


