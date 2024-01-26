import enum

import chess
import chess.pgn

CURRENT_POSSIBLE_MOVES: set[str] = None
CURRENT_BOARD: chess.Board = None


class ChessLanguage(str, enum.Enum):
	UCI = "UCI"
	SAN = "SAN"
	SAN_FRENCH = "SAN_FRENCH"


CURRENT_MODE: ChessLanguage = None


def validMove(move: str):
	return move.lower() in CURRENT_POSSIBLE_MOVES


def getCurrentPossibleMoves():
	"""
	Returns the list of possible moves for the current player in SAN format
	"""
	if CURRENT_MODE == ChessLanguage.SAN_FRENCH:
		movesSan = [CURRENT_BOARD.san(m) for m in CURRENT_BOARD.legal_moves]
		movesSanFrench = [sanToFrenchSan(m) for m in movesSan]
		return set(movesSanFrench)
	elif CURRENT_MODE == ChessLanguage.SAN:
		return set([CURRENT_BOARD.san(m) for m in CURRENT_BOARD.legal_moves])
	elif CURRENT_MODE == ChessLanguage.UCI:
		return set([CURRENT_BOARD.uci(m) for m in CURRENT_BOARD.legal_moves])


def sanToFrenchSan(m):
	"""
	Converts a move in SAN format to a move in SAN_FRENCH format
	"""
	return m.replace("R", "T").replace("N", "C").replace("B", "F").replace("Q", "D").replace("K", "R")


def frenchSanToSan(m):
	"""
	Converts a move in SAN_FRENCH format to a move in SAN format
	"""
	return m.replace("R", "K").replace("T", "R").replace("C", "N").replace("F", "B").replace("D", "Q")


def getMoveFromLowerOne(moveLower: str):
	"""
	Returns the move in SAN format from the move in lower case
	"""
	moves = getCurrentPossibleMoves() if CURRENT_MODE != ChessLanguage.SAN_FRENCH else set([CURRENT_BOARD.san(m) for m in CURRENT_BOARD.legal_moves])
	check = (lambda m: sanToFrenchSan(m).lower() == moveLower) if CURRENT_MODE == ChessLanguage.SAN_FRENCH else (lambda m: m.lower() == moveLower)
	for move in moves:
		if check(move):
			return move
	raise ValueError("Move not found")


def doMove(moveLower: str):
	"""
	Performs the move on the current board
	"""
	global CURRENT_POSSIBLE_MOVES
	move = getMoveFromLowerOne(moveLower)
	CURRENT_BOARD.push_san(move)
	CURRENT_POSSIBLE_MOVES = [m.lower() for m in getCurrentPossibleMoves()]


def init(language: ChessLanguage = ChessLanguage.SAN_FRENCH):
	global CURRENT_BOARD, CURRENT_POSSIBLE_MOVES, CURRENT_MODE
	CURRENT_MODE = language
	CURRENT_BOARD = chess.Board()
	CURRENT_POSSIBLE_MOVES = [m.lower() for m in getCurrentPossibleMoves()]


def getPGN():
	"""
	Returns the PGN string of the current board
	"""
	return chess.pgn.Game.from_board(CURRENT_BOARD).accept(chess.pgn.StringExporter())
