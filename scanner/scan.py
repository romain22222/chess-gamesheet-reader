import cv2
from imutils import perspective, resize, grab_contours
import numpy as np
from rembg.bg import remove as rembg
from PIL import Image

APPROX_POLY_DP_ACCURACY_RATIO = 0.02
IMG_RESIZE_H = 500.0


def scan(data: Image.Image) -> Image.Image:
	# add transparency channel
	data = data.convert("RGBA")
	img = cv2.cvtColor(np.array(rembg(data)), cv2.COLOR_RGBA2BGRA)
	orig = img.copy()

	ratio = img.shape[0] / IMG_RESIZE_H

	img = resize(img, height=int(IMG_RESIZE_H))
	_, img = cv2.threshold(img[:, :, 3], 0, 255, cv2.THRESH_BINARY)
	img = cv2.medianBlur(img, 15)

	cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	outline = None

	for c in cnts:
		perimeter = cv2.arcLength(c, True)
		polygon = cv2.approxPolyDP(c, APPROX_POLY_DP_ACCURACY_RATIO * perimeter, True)

		if len(polygon) == 4:
			outline = polygon.reshape(4, 2)

	if outline is None:
		r = orig
	else:
		r = perspective.four_point_transform(orig, outline * ratio)
	img = Image.fromarray(cv2.cvtColor(r, cv2.COLOR_BGRA2RGBA))
	return img

