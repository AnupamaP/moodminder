import sys, random, json, tweepy, time
from tweepy.streaming import StreamListener


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):
	def __init__(self, twitter_handle, classifier, view):
			StreamListener.__init__(self)
			self.twitter_handle = twitter_handle
			self.classifier = classifier
			self.view = view

	def on_data(self, data):
		data_dict = json.loads(data)
		if data_dict['user']['screen_name'] != self.twitter_handle:
				return True
		#blob = tb(data_dict['text'])
		#print blob, blob.sentiment.classification
		#print (self.classifier(data_dict['text']))
		#self.view(tweet+' '+str(self.classifier(tweet)))
		tweet = data_dict['text']
		classification = u'\U0001f61f'
		if self.classifier(tweet):
			classification = u'\U0001f604'
		self.view({'name': self.twitter_handle, 'tweets': tweet, 
					   'sentiment': classification, 'recommendation':'Be happy'})
		return True

	def on_error(self, status):
		print status