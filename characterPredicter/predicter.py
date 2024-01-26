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
	img = img.resize((64, 64))
	img = np.array(img)
	img = img.reshape(1, -1)
	predictions = loaded_model.predict_proba(img)[0]
	preds = [(labels[i], predictions[i]) for i in range(len(predictions))]
	preds.sort(key=lambda x: x[1], reverse=True)
	return preds[:5]
