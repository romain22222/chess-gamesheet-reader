import os
import shutil

import numpy as np
from PIL import Image
import cv2

rightParams = [
	(47, 5),
	(31, 5)
]


def extract_characters(image: Image.Image, index: int):
	imbytes = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

	# Convert the image to grayscale
	gray = cv2.cvtColor(imbytes, cv2.COLOR_BGR2GRAY)

	# # Save the grayscale image
	# cv2.imwrite("tmp/gray.png", gray)

	if not os.path.exists("tmp/charTest"):
		os.mkdir("tmp/charTest")
	# Create a small film with the adaptive thresholding while changing the block size
	# for i in range(41, 151, 2):
	# 	for j in range(0, 10):
	# 		tmp = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, i, j)
	# 		cv2.imwrite(f"tmp/charTest/name_{index}_{i}_{j}.png", tmp)
	#
	# toSelectI = input("Select the best i: ")
	# toSelectJ = input("Select the best j: ")
	toSelectI, toSelectJ = rightParams[0]
	shutil.rmtree("tmp/charTest")
	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, int(toSelectI), int(toSelectJ))

	retT = gray

	# Save the thresholded image
	cv2.imwrite("tmp/return.png", retT)
	cv2.imwrite("tmp/thresh.png", thresh)

	# Find contours
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Function to sort contours from left to right
	bounding_boxes = [cv2.boundingRect(c) for c in contours]
	(contours, _) = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0], reverse=False))
	charBB = []
	for idx, contour in enumerate(contours):
		# Get the rectangle that contains the contour
		x, y, w, h = cv2.boundingRect(contour)
		# Don't plot small false positives that aren't text
		if (w < 10 and h < 10) or w > 70 or w < 5:
			continue
		# Don't take into account the left most part of the image if index is multiple of 2
		if index % 2 == 0 and x < 50:
			continue
		charBB.append((x, y, w, h))

	charRet = []
	for x, y, w, h in charBB:
		charRet.append(cv2.cvtColor(retT[y:y+h, x:x+w], cv2.COLOR_GRAY2RGB))
	# On each character, do an otsu thresholding and transform it into a PIL image
	charRet = [Image.fromarray(cv2.cvtColor(cv2.threshold(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], cv2.COLOR_GRAY2RGB)) for c in charRet]
	[c.save(f"tmp/croppedC/char_{i}.png") for i, c in enumerate(charRet)]
	return charRet
