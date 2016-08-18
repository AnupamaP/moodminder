import sys

class PrintView(object):
	def __init__(self):
			self.view_out = []

	def updateView(self, out):
		#if len(self.view_out):
				#del(self.view_out[0])
		self.view_out.append(out)
		print 'after streaming?: ', out
	
	def getView(self):
		return self.view_out			