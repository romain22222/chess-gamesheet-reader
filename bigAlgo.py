import itertools

import numpy as np
from PIL import Image

import caseSplitter.splitter
import characterPredicter.predicter
from chessChecker import checker
from scanner.scan import scan


def getMoveFromUser(availableMoves=None):
	if availableMoves is None:
		availableMoves = checker.getCurrentPossibleMoves()
		prompt = "Cannot find the move. Please enter it"
	else:
		prompt = "Which move is this? ({})".format(", ".join([value[0] for value in availableMoves]))
	while True:
		move = input(prompt + " (E if game ended here, A to abort and exit the program): ")
		if move in availableMoves or move == "E":
			return move
		elif move == "A":
			abort = input("Are you sure you want to abort the notation? (y/n): ")
			if abort == "y":
				exit(0)
		prompt = "Invalid move. Please enter a valid move"


def imageToPGN(image):
	"""
	Converts an image to a PGN string
	"""
	# DONE ------------------------ Step 1: Scan the image and get the cropped game sheet
	imageScanned = scan(image)
	# Step 2: Cut the image's move cases
	moveCases = []
	result = None
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
		if move == "E":
			# It means that the game ended by an oral agreement
			# Ask the user what the result is
			while result not in range(4):
				result = input("What is the result of the game? (0: white wins, 1: draw, 2: black wins, 3: unknown): ")
			result = ["1-0", "1/2-1/2", "0-1", "*"][result]
			break
		# Add the move to the game
		checker.doMove(move)
		if checker.CURRENT_BOARD.is_game_over():
			break

	# Step 6: Recompose the PGN string and return it
	return checker.getPGN(result)


if __name__ == '__main__':
	checker.init(checker.ChessLanguage.SAN_FRENCH)
	image = Image.open("tmp/image1_scanned.jpg")
	splitted = caseSplitter.splitter.split(image)
	for i in range(len(splitted)):
		splitted[i].save(f"tmp/cropping/image1_scanned_cropped_{i}.jpg")
	# save the scanned image

# TODO
"""
Cut the image's move cases
Slice each move case character by character
"""
