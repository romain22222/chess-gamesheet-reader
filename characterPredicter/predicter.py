import pickle

loaded_model = pickle.load(open('characterPredicter/model.sav', 'rb'))


def predictTop5(data):
	"""
	Predicts the 5 most likely characters from the image
	:param data:
	:return: list of tuples (char, proba)
	"""
	predictions = loaded_model.predict_proba(data)
	predictions.sort(key=lambda x: x[1], reverse=True)
	return [(loaded_model.classes_[predictions[i][0]], predictions[i][1]) for i in range(5)]
