import numpy as np
from PIL import Image

import caseSplitter.splitter
import characterExtracter.extracter
import characterPredicter.predicter
from chessChecker import checker
from scanner.scan import scan


def getMoveFromUser(availableMovesOutside: tuple[str, float] = None) -> tuple[str, float]:
	availableMoves = checker.getCurrentPossibleMoves()
	print(availableMoves)
	if availableMovesOutside is None:
		prompt = "Cannot find the move. Please enter it"
	else:
		prompt = "Which move is this? ({})".format(", ".join([value[0] for value in availableMovesOutside]))
	while True:
		move = input(prompt + " (E if game ended here, A to abort and exit the program): ")
		if move in availableMoves or move == "E":
			return [move, 1.0]
		elif move == "A":
			abort = input("Are you sure you want to abort the notation? (y/n): ")
			if abort == "y":
				exit()
		prompt = "Invalid move. Please enter a valid move"


def askUserRightMove(move: tuple[str, float]) -> tuple[str, float]:
	while True:
		answer = input(f"Is this move ({move[0]}) correct? (y/n): ")
		if answer == "y":
			return move
		elif answer == "n":
			return getMoveFromUser()
		else:
			print("Invalid answer. Please enter 'y' or 'n'")


def imageToPGN(startingImagePath: str) -> str:
	"""
	Converts an image to a PGN string
	"""
	# Step 1: Scan the image and get the cropped game sheet
	imageScanned = scan(Image.open(startingImagePath))
	imageScanned.save("tmp/scan.png")
	# Step 2: Cut the image's move cases
	moveCases = caseSplitter.splitter.split(imageScanned)
	[moveCases[i].save("tmp/cropping/moveCase{}.png".format(i)) for i in range(len(moveCases))]
	result = None
	# For each move case:
	for moveCase in moveCases:
		# Step 3: Slice each move case character by character
		chars = characterExtracter.extracter.extract_characters(moveCase, moveCases.index(moveCase))
		# Step 4: Predict each character with the model
		# Each character is an image that needs to be resized to 64x64. The model will predict the 5 most likely characters
		predictions = []
		for char in chars:
			predictions.append(characterPredicter.predicter.predictTop5(char))
		# predictions of form [[(char, proba), ...], ...]
		# Create the "moves" list sorted by the most likely combination of characters
		print(predictions)
		moves = []
		for i in range(5 ** len(predictions)):
			indexes = [i // (5 ** j) % 5 for j in range(len(predictions))]
			if any([len(predictions[j]) <= indexes[j] for j in range(len(predictions))]):
				continue
			globalMove = "".join([predictions[j][indexes[j]][0] for j in range(len(predictions))])
			if checker.validMove(globalMove):
				moves.append(
					(globalMove, np.prod([predictions[j][i // (5 ** j) % 5][1] for j in range(len(predictions))])))

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
			else:
				# Ask the user if the move is correct, if not, prompt the user to enter which move it is
				move = askUserRightMove(move)
		if move == "E":
			# It means that the game ended by an oral agreement
			# Ask the user what the result is
			while result not in range(4):
				result = input("What is the result of the game? (0: white wins, 1: draw, 2: black wins, 3: unknown): ")
			result = ["1-0", "1/2-1/2", "0-1", "*"][result]
			break
		move = (move[0].lower(), move[1])
		print(move)
		# Add the move to the game
		checker.doMove(move[0])
		if checker.CURRENT_BOARD.is_game_over():
			break

	# Step 6: Recompose the PGN string and return it
	return checker.getPGN(result)


if __name__ == '__main__':
	checker.init(checker.ChessLanguage.SAN_FRENCH)
	print(imageToPGN("testSheets/image3.jpg"))

# TODO
"""
Slice each move case character by character
"""
