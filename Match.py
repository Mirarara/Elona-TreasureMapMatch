import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageGrab
a = ImageGrab.grabclipboard()
a.save('test.png','PNG')

# File to read for world map.
mapname = 'North_Tyris_Elona_Plus.webp'

img = cv.imread(mapname,0)
imgc = cv.imread(mapname)
img2 = img.copy()
template = cv.imread('test.png',0)
w, h = template.shape[::-1]

methods = ['cv.TM_CCOEFF_NORMED']
for meth in methods:
	img = img2.copy()
	method = eval(meth)
	# Apply template Matching
	res = cv.matchTemplate(img,template,method)
	min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

	top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)
	
	img = imgc.copy()
	cv.rectangle(img,top_left, bottom_right, (0,0,255), 5)
	plt.imshow(img[...,::-1])
	#plt.title('Detected Point'), 
	plt.xticks([]), plt.yticks([])
	plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
	plt.show()