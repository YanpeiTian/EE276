# Write images into patches
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np
import json
import os
import constant as const

def patch_write(img_name):
	img = mpimg.imread(img_name)

	# create directory
	path = os.getcwd() + '\\' + 'test'
	try:
		os.mkdir(path)
	except OSError:
		print ("Creation of the directory %s failed" % path)

	# Writing image info
	data = {}
	data[const.NAME] = img_name
	data[const.SIZE] = img.shape
	with open(path + '\\' + 'data.txt', 'w') as outfile:
		json.dump(data, outfile)

	h_filled = img.shape[0]//const.PATCH_SIZE[0] + 1
	w_filled = img.shape[1]//const.PATCH_SIZE[1] + 1
	img_filled = np.array()


def main():
	# Read Images 
	img = mpimg.imread('abigail-keenan-27293.png') 

	# Write to patches
	patch_write('abigail-keenan-27293.png')

	# Output Images 
	plt.imshow(img)
	# plt.show()


if __name__ == "__main__":
    main()

