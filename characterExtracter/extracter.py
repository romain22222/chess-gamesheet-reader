import numpy as np
from PIL import Image
import cv2


def extract_characters(image: Image.Image):
	imbytes = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

	# Convert the image to grayscale
	gray = cv2.cvtColor(imbytes, cv2.COLOR_BGR2GRAY)

	# # Save the grayscale image
	# cv2.imwrite("tmp/gray.png", gray)

	# Apply thresholding to get a binary image
	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 251, 2)
	retT = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 251, 2)

	# # Save the thresholded image
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
		if (w < 15 and h < 15) or w > 70 or x < 60:
			continue
		charBB.append((x, y, w, h))

	charRet = []
	newBB = []
	for x, y, w, h in charBB:
		iToRemove = []
		# If any box is overlapped in the y direction, merge it with the previous box
		for i, dims in enumerate(newBB):
			x2, y2, w2, h2 = dims
			if y2 + 10 < y + h < y2 + h2 - 10 or y2 + 10 < y < y2 + h2 - 10:
				iToRemove.append(i)
				x, y, w, h = min(x, x2), min(y, y2), max(x + w, x2 + w2) - min(x, x2), max(y + h, y2 + h2) - min(y, y2)
		if len(iToRemove) > 0:
			# Remove the boxes that were merged
			for i in sorted(iToRemove, reverse=True):
				del newBB[i]
		newBB.append((x, y, w, h))
	for x, y, w, h in newBB:
		charRet.append(Image.fromarray(cv2.cvtColor(retT[y:y+h, x:x+w], cv2.COLOR_GRAY2RGB)))
	[c.save(f"tmp/croppedC/char_{i}.png") for i, c in enumerate(charRet)]
	return charRet
