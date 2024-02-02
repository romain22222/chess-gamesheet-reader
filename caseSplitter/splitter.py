import math
from itertools import groupby, count

import cv2
import numpy as np
from PIL import Image


def intersection(line1, line2):
	"""
	:param line1: (rho, theta)
	:param line2: (rho, theta)
	:return: (x, y)
	"""
	rho1, theta1 = line1
	rho2, theta2 = line2
	A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
	b = np.array([rho1, rho2])
	x0, y0 = np.linalg.solve(A, b)
	return x0, y0


def splitV1(image: Image.Image):
	imbytes = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

	# Convert the image to grayscale
	gray = cv2.cvtColor(imbytes, cv2.COLOR_BGR2GRAY)

	# Apply GaussianBlur to reduce noise
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	# Perform edge detection using Canny
	edges = cv2.Canny(blurred, 50, 150)
	lines = cv2.HoughLines(edges, 1, np.pi / 180, 210, min_theta=np.pi / 2 - 0.05, max_theta=np.pi / 2 + 0.05)
	meanHeights = []
	allHeights = []
	for line in lines:
		rho, theta = line[0]
		if not any([abs(rho - r) < 20 for r in [h[0] for h in meanHeights]]) and rho > image.size[1] / 5.5:
			meanHeights.append((rho, theta))
		allHeights.append((rho, theta))

	meanHeights.sort()

	medianDiff = np.median([meanHeights[i + 1][0] - meanHeights[i][0] for i in range(len(meanHeights) - 1)])
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
		availableLines = [line for line in allHeights if
						  abs(line[0] - chosenLines[-1][0] - medianDiff * (skipped + 1)) < medianDiff / 2]
		# print(f"Found {len(availableLines)} lines: {availableLines}")
		if len(availableLines) == 0:
			unavailableLines.append(i)
			skipped += 1
			continue
		skipped = 0
		chosenLines.append(availableLines[0])
	# Get ranges of unavailable lines (ie if i have 2,3,4,6, then i have 2-4 and 6)
	unavailableLines = [list(g) for _, g in groupby(unavailableLines, lambda n, c=count(): n - next(c))]
	for r in unavailableLines:
		bottom = chosenLines[r[0] - 1]
		top = chosenLines[r[0]] if r[0] < len(chosenLines) else chosenLines[r[0] - 1] + medianDiff
		diffToAdd = (top[0] - bottom[0]) / (len(r) + 1)
		# print(f"Adding {diffToAdd} between {r[0]} and {r[-1]}")
		for i in range(r[0], r[-1] + 1):
			chosenLines.insert(i, (bottom[0] + diffToAdd * (i - r[0] + 1), (bottom[1] + top[1]) / 2))
	croppedImages = []
	for i in range(1, len(chosenLines) - 1):
		croppedImages.append(image.crop((0, chosenLines[i][0]-5, image.size[0]//4, chosenLines[i + 1][0]+10)))
		croppedImages.append(image.crop((image.size[0]//4, chosenLines[i][0]-5, image.size[0]//2, chosenLines[i + 1][0]+10)))
	for i in range(1, len(chosenLines) - 1):
		croppedImages.append(image.crop((image.size[0]//2, chosenLines[i][0]-5, image.size[0]//4*3, chosenLines[i + 1][0]+10)))
		croppedImages.append(image.crop((image.size[0]//4*3, chosenLines[i][0]-5, image.size[0], chosenLines[i + 1][0]+10)))
	return croppedImages


def splitV2(image: Image.Image):
	imbytes = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

	# Convert the image to grayscale
	gray = cv2.cvtColor(imbytes, cv2.COLOR_BGR2GRAY)

	# Apply GaussianBlur to reduce noise
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	# Perform edge detection using Canny
	edges = cv2.Canny(blurred, 50, 150)

	# First detect vertical lines and split the image along the 3 most common vertical lines in 4 parts
	lines = cv2.HoughLines(edges, 1, np.pi / 180, 360, min_theta=np.pi - 0.05, max_theta=np.pi + 0.05)
	mainLines = []
	for line in lines:
		rho, theta = line[0]
		mainLines.append((rho, theta))

	mainLines.sort(key=lambda x: x[0])
	keepLines = []
	for v in mainLines:
		if len(keepLines) == 0:
			keepLines.append(v)
		else:
			if abs(keepLines[-1][0] - v[0]) > image.size[0] / 8:
				keepLines.append(v)
	keepLines.sort(key=lambda x: intersection(x, (0, np.pi / 2))[0])
	# For each keepLine get the mean of the thetas / rhos in the mainLines that are close to the keepLine
	for i, keepLine in enumerate(keepLines):
		# Get the lines that are close to the keepLine
		closeLines = []
		for line in mainLines:
			if abs(line[0] - keepLine[0]) < image.size[0] / 8:
				closeLines.append(line)
		# Get the mean of the thetas / rhos
		keepLines[i] = (
		sum([l[0] for l in closeLines]) / len(closeLines), sum([l[1] for l in closeLines]) / len(closeLines))
	# Add the first and last lines
	keepLines.insert(0, (0, keepLines[0][1]))
	keepLines.append((2*keepLines[-1][0]-keepLines[-2][0], keepLines[-1][1]))
	# Do the same for horizontal lines
	lines = cv2.HoughLines(edges, 1, np.pi / 180, 210, min_theta=np.pi / 2 - 0.05, max_theta=np.pi / 2 + 0.05)
	meanHeights = []
	allHeights = []
	for line in lines:
		rho, theta = line[0]
		if not any([abs(rho - r) < 20 for r in [h[0] for h in meanHeights]]) and rho > image.size[1] / 5.5:
			meanHeights.append((rho, theta))
		allHeights.append((rho, theta))

	meanHeights.sort()

	medianDiff = np.median([meanHeights[i + 1][0] - meanHeights[i][0] for i in range(len(meanHeights) - 1)])
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
		availableLines = [line for line in allHeights if
						  abs(line[0] - chosenLines[-1][0] - medianDiff * (skipped + 1)) < medianDiff / 2]
		# print(f"Found {len(availableLines)} lines: {availableLines}")
		if len(availableLines) == 0:
			unavailableLines.append(i)
			skipped += 1
			continue
		skipped = 0
		chosenLines.append(availableLines[0])
	# print(unavailableLines)
	# print(medianDiff)
	# Get ranges of unavailable lines (ie if i have 2,3,4,6, then i have 2-4 and 6)
	unavailableLines = [list(g) for _, g in groupby(unavailableLines, lambda n, c=count(): n - next(c))]
	for r in unavailableLines:
		bottom = chosenLines[r[0] - 1]
		top = chosenLines[r[0]] if r[0] < len(chosenLines) else chosenLines[r[0] - 1] + medianDiff
		diffToAdd = (top[0] - bottom[0]) / (len(r) + 1)
		# print(f"Adding {diffToAdd} between {r[0]} and {r[-1]}")
		for i in range(r[0], r[-1] + 1):
			chosenLines.insert(i, (bottom[0] + diffToAdd * (i - r[0] + 1), (bottom[1] + top[1]) / 2))

	croppedImages = []
	# Recolt the images delimited by the polygons formed by the chosenLines and the keepLines
	for j in range(1, len(chosenLines) - 1):
		for i in range(len(keepLines) - 1):
			topLine = chosenLines[j]
			bottomLine = chosenLines[j + 1]
			topLine = (topLine[0] - 5, topLine[1])
			bottomLine = (bottomLine[0] + 5, bottomLine[1])
			# Consider the polygon must be a rectangle, perform a perspective transform to get the rectangle
			pts1 = np.float32([
				intersection(keepLines[i], topLine), intersection(keepLines[i + 1], topLine),
				intersection(keepLines[i], bottomLine), intersection(keepLines[i + 1], bottomLine)])
			pts2 = np.float32([
				[0, 0], [image.size[0]//4, 0], [0, medianDiff], [image.size[0]//4, medianDiff]])
			matrix = cv2.getPerspectiveTransform(pts1, pts2)
			# Save imbytes
			result = cv2.warpPerspective(imbytes, matrix, (image.size[0]//4, math.ceil(medianDiff)))
			croppedImages.append(Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)))
	# Rearrange the image so that for each 4 images, the 1st 2 are in a first list and the 2nd 2 in a second list
	croppedImages = [croppedImages[cI] for cI in range(len(croppedImages)) if cI % 4 < 2] + [croppedImages[cI] for cI in range(len(croppedImages)) if cI % 4 >= 2]
	return croppedImages


def split(image: Image.Image):
	# return splitV1(image)
	return splitV2(image)
