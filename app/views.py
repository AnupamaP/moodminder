# views.py 

from django.shortcuts import render, HttpResponse, render_to_response
import requests
import sys, os, time
import TweetView
import tweepy, json, TweetProcess, TweetClassify, TweetView, scipy
import threading

#cur_dir = os.getcwd()
#sys.path.append(cur_dir+"/app/code")

# Create your views here.

# Global data structure
view = TweetView.PrintView()

def index(request):
    return HttpResponse('Hello World!')

def test(request):
    return HttpResponse('My second view!')

def profile_old(request):
	import tweepy, json
	text = " * "
	parsedData = []
	if request.method == 'POST':
		handle = request.POST.get('user')
		access_token = "716775869046513664-5A4VLHy6O2AlxgPYnnDqK68hUgovsea"
		access_token_secret = "9imA5sFl6koFoweWeDbMzti0EwQs3b2gb5JszsAic4DSB"
		consumer_key = "XTb6wIyh9M2ZXaeNgIQcZ52JX"
		consumer_secret = "6XLYcm4YT2aAqxEpd9Ym4AZaW12rBRNsoXHNRwvLW3g1hG0L5B"

		auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
		auth.set_access_token(access_token, access_token_secret)

		api = tweepy.API(auth)
		tweets = api.user_timeline(screen_name = handle, count = 1, include_rts = True)
		
		userData = {}
		userData['name'] = handle
		userData['tweets'] = tweets[0].text
		userData['sentiment'] = ":)"
		parsedData.append(userData)
		print parsedData
	return render(request, 'app/profile.html', {'data': parsedData})
	
def wait_page(request, twitter_handle):
	print 'HERE>>>'
	tp = TweetProcess.TweetProcess(twitter_handle, TweetClassify.NaiveBayesClassifier(), view.updateView)
	tp.tweet_process(1, request) # 0=past tweet, 1=stream
	return True

def profile_stream(request):
	if request.method == 'POST':
		twitter_handle = request.POST.get('user')
		#tp = TweetProcess.TweetProcess(twitter_handle, TweetClassify.NaiveBayesClassifier(), view.updateView)
		#userData = tp.tweet_process(stream=0)
		t = threading.Thread(target=wait_page, args=(request, twitter_handle,))
		t.start()
		#tp.tweet_process(1, request) # 0=past tweet, 1=stream
		#time.sleep(10)
		#parsedData.append(userData)
		#print parsedData
		return render(request, 'app/profile_stream.html', {'data': view.getView()})
	elif request.method == 'GET':
		#parsedData.append(userData)
		#print parsedData
		return render(request, 'app/profile_stream.html', {'data': view.getView()})	
	else:
		return render(request, 'app/profile_stream.html', {'data': view.getView()})
		
def profile_past(request):
	if request.method == 'POST':
		twitter_handle = request.POST.get('user')
		tp = TweetProcess.TweetProcess(twitter_handle, TweetClassify.NaiveBayesClassifier(), view.updateView)
		tp.tweet_process(0, request) # 0=past tweet, 1=stream
		#parsedData.append(userData)
		#print parsedData
		return render(request, 'app/profile_past.html', {'data': view.getView()})
	elif request.method == 'GET':
		#parsedData.append(userData)
		#print parsedData
		return render(request, 'app/profile_past.html', {'data': view.getView()})	
	else:
		return render(request, 'app/profile_past.html', {'data': view.getView()})	
		
		