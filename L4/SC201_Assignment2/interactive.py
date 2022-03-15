"""
File: interactive.py
Name: Cheng-Chu Chung
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""
from submission import *


def main():
	"""
	although this movie suffers from some cliche , this film is still worth watching
	:return: dataset, str
	"""
	weights = {}
	with open('weights', encoding="utf-8") as f:
		for line in f:
			line = line.strip().split()
			weights[line[0]] = float(line[1])
	interactivePrompt(extractWordFeatures, weights)


if __name__ == '__main__':
	main()