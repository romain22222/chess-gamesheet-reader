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
	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 161, 2)
	retT = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 161, 2)

	# # Save the thresholded image
	# cv2.imwrite("tmp/thresh.png", thresh)

	# Find contours
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Function to sort contours from left to right
	bounding_boxes = [cv2.boundingRect(c) for c in contours]
	(contours, _) = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0], reverse=False))
	charRet = []
	for idx, contour in enumerate(contours):
		# Get the rectangle that contains the contour
		x, y, w, h = cv2.boundingRect(contour)
		# Don't plot small false positives that aren't text
		if (w < 15 and h < 15) or w > 70 or x < 60:
			continue
		# Save the cropped image
		charRet.append(Image.fromarray(cv2.cvtColor(retT[y:y+h, x:x+w], cv2.COLOR_GRAY2RGB)))
	return charRet
