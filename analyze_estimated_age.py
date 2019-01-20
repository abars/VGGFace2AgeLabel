# ----------------------------------------------
# Analyze estimated data
# ----------------------------------------------

import cv2
import sys
import numpy as np
import os

import matplotlib.pyplot as plt

lines=open("./estimated/identity_meta_with_estimated_age.csv").readlines()

fig = plt.figure()
ax1 = fig.add_axes((0.1, 0.6, 0.8, 0.3))
ax2 = fig.add_axes((0.1, 0.1, 0.8, 0.3))
ax1.tick_params(labelbottom="on")
ax2.tick_params(labelleft="on")

max_cnt=len(lines)

gender_list=np.zeros((max_cnt))
age_list=np.zeros((max_cnt))

DISTRIBUTION_FILE='./estimated/estimated_distribution.png'

cnt=0

for line in lines:
	obj=line.split(", ")
	path=obj[0]
	trainset=obj[3]
	gender=obj[4].strip()
	age=int(obj[5].strip())

	if gender=="f":
		gender_list[cnt]=0
	else:
		gender_list[cnt]=1
	age_list[cnt]=age

	cnt=cnt+1

ax1.hist(gender_list, bins=2)
ax1.set_title('gender')
ax1.set_xlabel('gender')
ax1.set_ylabel('count')
ax1.legend(loc='upper right')

ax2.hist(age_list, bins=101, range=(0,100))
ax2.set_title('age')
ax2.set_xlabel('age')
ax2.set_ylabel('count')
ax2.legend(loc='upper right')

fig.savefig(DISTRIBUTION_FILE)
sys.exit(1)

