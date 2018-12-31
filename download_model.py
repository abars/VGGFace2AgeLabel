#Download pretrained model

import os
import sys
if sys.version_info >= (3,0):
	from urllib import request
else:
	import urllib2

def main(argv):
	OUTPUT_PATH="./pretrain/"
	if not os.path.isdir(OUTPUT_PATH):
		os.mkdir(OUTPUT_PATH)
	print("1/2");
	with open(OUTPUT_PATH+'dex_imdb_wiki.caffemodel','wb') as f:
		path="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_imdb_wiki.caffemodel"
		if sys.version_info >= (3,0):
			f.write(request.urlopen(path).read())
		else:
			f.write(urllib2.urlopen(path).read())
		f.close()
	print("2/2");
	with open(OUTPUT_PATH+'age.prototxt5','wb') as f:
		path="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt"
		if sys.version_info >= (3,0):
			f.write(request.urlopen(path).read())
		else:
			f.write(urllib2.urlopen(path).read())
		f.close()

if __name__=='__main__':
	main(sys.argv[1:])
