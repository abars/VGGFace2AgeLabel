# ----------------------------------------------
# Predict age
# ----------------------------------------------

import cv2
import sys
import numpy as np
import os
import caffe
import glob

# ----------------------------------------------
# MODE
# ----------------------------------------------

DATASET_ROOT_PATH="/Volumes/TB4/Keras/"

# ----------------------------------------------
# Argument
# ----------------------------------------------

#DATASET_ROOT_PATH=""

if len(sys.argv) >= 1:
  if len(sys.argv) >= 2:
    DATASET_ROOT_PATH=sys.argv[3]
else:
  print("usage: python calculate_age.py [datasetroot(optional)]")
  sys.exit(1)

# ----------------------------------------------
# Load Model
# ----------------------------------------------

net  = caffe.Net('pretrain/age.prototxt', 'pretrain/dex_imdb_wiki.caffemodel', caffe.TEST)

def calc_age(image):
	img = cv2.imread(image)

	image_size=224

	img = cv2.resize(img, (image_size, image_size))
	img = img[...,::-1]  #BGR 2 RGB

	data = np.array(img, dtype=np.float32)
	data.shape = (1,) + data.shape

	data -= 128

	data = data.transpose((0, 3, 1, 2))
	out = net.forward_all(data = data)
	pred = out[net.outputs[0]]

	age=0
	for i in range(0,101):
		age=age+pred[0][i]*i

	return age

# ----------------------------------------------
# FileList
# ----------------------------------------------

lines=open(DATASET_ROOT_PATH+"Dataset/identity_meta.csv").readlines()

OUTPUT_PATH="./identity_meta_with_estimated_age.csv"
if(os.path.exists(OUTPUT_PATH)):
	output_data=open(OUTPUT_PATH).readlines()	#for continue
else:
	output_data=[]

cache={}
for line in output_data:
	obj=line.split(", ")
	path=obj[0]
	age=obj[5].strip()
	cache[path]=age

DEBUG_MODE=1

with open(OUTPUT_PATH, mode='w') as f:
	for line in lines:
		obj=line.split(", ")
		path=obj[0]
		trainset=obj[3]

		print("target person : "+path)

		if path in cache.keys():
			#already estimated
			f.write(line.strip()+", "+str(cache[path])+"\n")
			print("  use cached age "+str(cache[path]))
		else:
			#yet estimated
			if trainset=="0":# or trainset=="1":
				if trainset=="0":
					path2=DATASET_ROOT_PATH+"Dataset/test/"+path
				else:
					path2=DATASET_ROOT_PATH+"Dataset/train/"+path
				age_sum=0
				age_cnt=0
				for image_path in glob.glob(path2+"/*.jpg"):
					age=calc_age(image_path)
					age_sum+=age
					age_cnt=age_cnt+1
					print(" path : "+image_path+" age : "+str(age))
					if DEBUG_MODE==1:
						break
				age=int(round(age_sum/age_cnt))
				print("  avarage age : "+str(age))
				f.write(line.strip()+", "+str(age)+"\n")



