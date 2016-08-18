# app/urls.py

from django.conf.urls import url

from app import views


urlpatterns = [
	url(r'^$', views.index, name='index'),
	url(r'^test/$', views.test, name='test'),
	url(r'^profile_past/$', views.profile_past, name='profile_past'),
	url(r'^profile_stream/$', views.profile_stream, name='profile_stream'),
]