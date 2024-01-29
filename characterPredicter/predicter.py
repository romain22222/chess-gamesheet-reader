import pickle

import numpy as np
from PIL import Image

loaded_model = pickle.load(open('characterPredicter/model.sav', 'rb'))

labels = dict([(i, loaded_model.classes_[i]) for i in range(len(loaded_model.classes_))])


def predictTop5(img: Image.Image):
	"""
	Predicts the 5 most likely characters from the image
	:param img:
	:return: list of tuples (char, proba)
	"""
	img = img.convert("L")
	preds = []
	for i in range(5):
		tmp = Image.new("L", (img.width + 2, img.height + 2), 0)
		tmp.paste(img, (1, 1))
		img = tmp
		checkingImage = img.resize((64, 64))
		checkingImage = np.array(checkingImage)
		checkingImage = checkingImage.reshape(1, -1)
		predictions = loaded_model.predict_proba(checkingImage)[0]
		tmpPreds = [(labels[i], predictions[i]) for i in range(len(predictions))]
		tmpPreds.sort(key=lambda x: x[1], reverse=True)
		preds.append(tmpPreds[0])
	preds.sort(key=lambda x: x[1], reverse=True)
	return preds
