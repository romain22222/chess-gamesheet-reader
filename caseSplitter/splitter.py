from itertools import groupby, count

import cv2
import numpy as np
from PIL import Image


def split(image: Image):
	imbytes = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

	# Convert the image to grayscale
	gray = cv2.cvtColor(imbytes, cv2.COLOR_BGR2GRAY)

	# Apply GaussianBlur to reduce noise
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	# Perform edge detection using Canny
	edges = cv2.Canny(blurred, 50, 150)

	Image.fromarray(edges).save("tmp/edges.png")

	# Detect horizontal lines
	lines = cv2.HoughLines(edges, 1, np.pi/180, 210, min_theta=np.pi/2-0.05, max_theta=np.pi/2+0.05)
	meanHeights = []
	allHeights = []
	for line in lines:
		rho, theta = line[0]
		# a = np.cos(theta)
		b = np.sin(theta)
		# x0 = a * rho
		y0 = b * rho
		# x1 = int(x0 + 1000 * (-b))
		# y1 = int(y0 + 1000 * a)
		# x2 = int(x0 - 1000 * (-b))
		# y2 = int(y0 - 1000 * a)
		if not any([abs(y0 - y) < 20 for y in meanHeights]):
			# cv2.line(imbytes, (x1, y1), (x2, y2), (0, 0, 255), 2)
			meanHeights.append(y0)
		allHeights.append(y0)
	# Save the imbytes in tmp/image1_scanned_splitted.png
	Image.fromarray(cv2.cvtColor(imbytes, cv2.COLOR_BGR2RGB)).save("tmp/image1_scanned_splitted.png")
	meanHeights.sort()
	# print(meanHeights)
	medianDiff = np.median([meanHeights[i+1] - meanHeights[i] for i in range(len(meanHeights) - 1)])
	top = meanHeights[0]
	# You must have 40 cropped images
	# for each delta of medianDiff, find the line that is the mean of all the lines that are in the delta
	# if there is no line in the delta, then generate a line at the mean of the last line plus medianDiff
	unavailableLines = []
	chosenLines = [top]
	allHeights.sort()
	# print(medianDiff)
	skipped = 0
	for i in range(1, 42):
		availableLines = [line for line in allHeights if abs(line - chosenLines[-1] - medianDiff * (skipped+1)) < medianDiff/2]
		print(f"Found {len(availableLines)} lines: {availableLines}")
		if len(availableLines) == 0:
			unavailableLines.append(i)
			skipped += 1
			continue
		skipped = 0
		chosenLines.append(availableLines[0])
	# print(unavailableLines)
	# print(medianDiff)
	# Get ranges of unavailable lines (ie if i have 2,3,4,6, then i have 2-4 and 6)
	unavailableLines = [list(g) for _, g in groupby(unavailableLines, lambda n, c=count(): n-next(c))]
	for r in unavailableLines:
		bottom = chosenLines[r[0]-1]
		top = chosenLines[r[0]] if r[0] < len(chosenLines) else chosenLines[r[0]-1] + medianDiff
		diffToAdd = (top - bottom) / (len(r) + 1)
		print(f"Adding {diffToAdd} between {r[0]} and {r[-1]}")
		for i in range(r[0], r[-1] + 1):
			chosenLines.insert(i, bottom + diffToAdd * (i - r[0] + 1))
	# print(chosenLines)
	croppedImagesLines = []
	for i in range(len(chosenLines) - 1):
		croppedImagesLines.append(imbytes[int(chosenLines[i]-5):int(chosenLines[i+1]+10), :])
	croppedImages = []
	for i in range(1, len(croppedImagesLines)):
		croppedImages.append(croppedImagesLines[i][:, 0:croppedImagesLines[i].shape[1]//4])
		croppedImages.append(croppedImagesLines[i][:, croppedImagesLines[i].shape[1]//4:croppedImagesLines[i].shape[1]//2])
	for i in range(1, len(croppedImagesLines)):
		croppedImages.append(croppedImagesLines[i][:, croppedImagesLines[i].shape[1]//2:3*croppedImagesLines[i].shape[1]//4])
		croppedImages.append(croppedImagesLines[i][:, 3*croppedImagesLines[i].shape[1]//4:croppedImagesLines[i].shape[1]])
	# Returns an image from PIL
	return [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in croppedImages]
