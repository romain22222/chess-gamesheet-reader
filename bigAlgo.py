import itertools

import numpy as np

import characterPredicter.predicter
from chessChecker import checker
from scanner.scan import scan


def getMoveFromUser(availableMoves=None):
	if availableMoves is None:
		availableMoves = checker.getCurrentPossibleMoves()
		prompt = "Cannot find the move. Please enter it: "
	else:
		prompt = "Which move is this? ({}): ".format(", ".join([value[0] for value in availableMoves]))
	while True:
		move = input(prompt)
		if move in availableMoves:
			return move
		prompt = "Invalid move. Please enter a valid move: "


def imageToPGN(image):
	"""
	Converts an image to a PGN string
	"""
	# DONE ------------------------ Step 1: Scan the image and get the cropped game sheet
	imageScanned = scan(image)
	# Step 2: Cut the image's move cases
	moveCases = []
	# For each move case:
	for moveCase in moveCases:
		# Step 3: Slice each move case character by character
		chars = []
		# Step 4: Predict each character with the model
		# Each character is an image that needs to be resized to 64x64. The model will predict the 5 most likely characters
		predictions = []
		for char in chars:
			...
		# predictions of form [[(char, proba), ...], ...]
		# DONE ---------------------- Create the "moves" list sorted by the most likely combination of characters
		moves = []
		for comb in itertools.filterfalse(lambda x: not checker.validMove("".join([value[0] for value in x])),
										  itertools.product(predictions, repeat=len(chars))):
			moves.append("".join([value[0] for value in comb]), np.prod([value[1] for value in comb]))

		# DONE ----------------------- Step 5: Get the most likely move based on the predictions
		moves.sort(key=lambda x: x[1], reverse=True)
		if len(moves) == 0:
			# If no move is likely enough, prompt the user to enter the move
			move = getMoveFromUser()
		else:
			move = moves[0]
			if move[1] < 0.5:
				# If the move is not likely enough, prompt the user to enter which move it is
				move = getMoveFromUser(moves)
		# Add the move to the game
		checker.doMove(move)

	# Step 6: Recompose the PGN string and return it
	return checker.getPGN()


if __name__ == '__main__':
	checker.init(checker.ChessLanguage.SAN_FRENCH)
	predictions = characterPredicter.predicter.predictTop5(open("curated/100/6567.png", "rb").read())
	print(predictions)

# TODO
"""
Cut the image's move cases
Slice each move case character by character
"""
