# ----------------------------------------------
# Predict age
# ----------------------------------------------

import cv2
import sys
import numpy as np
import os
import caffe

# ----------------------------------------------
# MODE
# ----------------------------------------------

#DATASET_ROOT_PATH="/Volumes/TB4/Keras/"

# ----------------------------------------------
# Argument
# ----------------------------------------------

if len(sys.argv) >= 1:
  if len(sys.argv) >= 2:
    DATASET_ROOT_PATH=sys.argv[3]
else:
  print("usage: python calculate_age.py [datasetroot(optional)]")
  sys.exit(1)

image="image/60.jpg"
print image
img = cv2.imread(image)

image_size=224

img = cv2.resize(img, (image_size, image_size))
img = img[...,::-1]  #BGR 2 RGB

data = np.array(img, dtype=np.float32)
data.shape = (1,) + data.shape

data -= 128

# ----------------------------------------------
# Calculate age
# ----------------------------------------------

net  = caffe.Net('pretrain/age.prototxt', 'pretrain/dex_imdb_wiki.caffemodel', caffe.TEST)
data = data.transpose((0, 3, 1, 2))
out = net.forward_all(data = data)
pred = out[net.outputs[0]]

age=0
for i in range(0,101):
	age=age+pred[0][i]*i

print age

