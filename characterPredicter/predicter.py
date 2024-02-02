import pickle

import numpy as np
from PIL import Image

loaded_model = pickle.load(open('characterPredicter/model.sav', 'rb'))

labels = dict([(i, loaded_model.classes_[i] if loaded_model.classes_[i] != "0" else "O") for i in range(len(loaded_model.classes_))])

baseStoredPred = {key: 0 for key in labels.values()}

def predictTop5(img: Image.Image):
	"""
	Predicts the 5 most likely characters from the image
	:param img:
	:return: list of tuples (char, proba)
	"""
	img = img.convert("L")
	# Init the stored predictions
	storedPred = baseStoredPred.copy()
	for i in range(10):
		tmp = Image.new("L", (img.width + 2*i, img.height + 2*i), 0)
		tmp.paste(img, (i, i))
		tmp = tmp.resize((64, 64))
		tmp = np.array(tmp)
		tmp = tmp.reshape(1, -1)
		predictions = loaded_model.predict_proba(tmp)[0]
		# Store only the 1st and 2nd most likely characters
		preds = [(labels[i], predictions[i]) for i in range(len(predictions))]
		preds.sort(key=lambda x: x[1], reverse=True)
		for pred in preds:
			storedPred[pred[0]] += pred[1]
	# Sort the stored predictions
	storedPred = [(key, storedPred[key]) for key in storedPred]
	storedPred.sort(key=lambda x: x[1], reverse=True)
	return storedPred[:5]
