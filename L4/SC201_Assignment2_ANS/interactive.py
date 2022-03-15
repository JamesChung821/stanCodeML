"""
File: interactive.py
Name: 
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users through Console. Remember to read the weights
and build a Dict[str: float]
"""

from util import interactivePrompt
from submission import extractWordFeatures


def main():
	weights = {}
	with open('weights', 'r') as f:
		for line in f:
			key, value = line.split()
			weights[key] = float(value)
	interactivePrompt(extractWordFeatures, weights)


if __name__ == '__main__':
	main()