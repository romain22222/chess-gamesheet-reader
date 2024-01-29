import itertools

import numpy as np
from PIL import Image

import caseSplitter.splitter
import characterExtracter.extracter
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
			return [move, 1.0]
		elif move == "A":
			abort = input("Are you sure you want to abort the notation? (y/n): ")
			if abort == "y":
				exit()
		prompt = "Invalid move. Please enter a valid move"


def imageToPGN(startingImagePath: str):
	"""
	Converts an image to a PGN string
	"""
	# Step 1: Scan the image and get the cropped game sheet
	imageScanned = scan(Image.open(startingImagePath))
	# Step 2: Cut the image's move cases
	moveCases = caseSplitter.splitter.split(imageScanned)
	result = None
	# For each move case:
	for moveCase in moveCases:
		# Step 3: Slice each move case character by character
		chars = characterExtracter.extracter.extract_characters(moveCase)
		# Step 4: Predict each character with the model
		# Each character is an image that needs to be resized to 64x64. The model will predict the 5 most likely characters
		predictions = []
		for char in chars:
			predictions.append(characterPredicter.predicter.predictTop5(char))
		# predictions of form [[(char, proba), ...], ...]
		# Create the "moves" list sorted by the most likely combination of characters
		print(predictions)
		moves = []
		for i in range(5**len(predictions)):
			globalMove = "".join([predictions[j][i // (5**j) % 5][0] for j in range(len(predictions))])
			if checker.validMove(globalMove):
				moves.append((globalMove, np.prod([predictions[j][i // (5**j) % 5][1] for j in range(len(predictions))])))

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
		print(move)
		# Add the move to the game
		checker.doMove(move[0])
		if checker.CURRENT_BOARD.is_game_over():
			break

	# Step 6: Recompose the PGN string and return it
	return checker.getPGN(result)


if __name__ == '__main__':
	checker.init(checker.ChessLanguage.SAN_FRENCH)
	print(imageToPGN("testSheets/image2.jpg"))

# TODO
"""
Slice each move case character by character
"""
