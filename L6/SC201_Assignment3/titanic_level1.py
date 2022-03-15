"""
File: titanic_level1.py
Name: Cheng-Chu Chung
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python codes. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle. This model is the most flexible one among all
levels. You should do hyperparameter tuning and find the best model.
"""

import math
import numpy as np
import pandas as pd
from util import *

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating the mode we are using
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	if mode == 'Train':
		with open(filename, 'r') as f:
			first_row = True	 # A switch
			column_names = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']	 # Column notes
			for column_name in column_names:	 # Create columns in dictionary
				data[column_name] = []

			for line in f:
				if first_row:	 # Skip the firs column row
					first_row = False

				else:
					data_in_row = line.strip().split(',')
					if data_in_row[6] is not '' and data_in_row[12] is not '':		# Assign data only when it exists
						for index in range(len(data_in_row)):
							if index == 1:
								data['Survived'].append(int(data_in_row[1]))

							elif index == 2:
								data['Pclass'].append(int(data_in_row[2]))

							elif index == 5:
								if data_in_row[5] == 'male':
									data['Sex'].append(1)
								elif data_in_row[5] == 'female':
									data['Sex'].append(0)

							elif index == 6:
								data['Age'].append(float(data_in_row[6]))

							elif index == 7:
								data['SibSp'].append(int(data_in_row[7]))

							elif index == 8:
								data['Parch'].append(int(data_in_row[8]))

							elif index == 10:
								data['Fare'].append(float(data_in_row[10]))

							elif index == 12:
								if data_in_row[12] == 'S':
									data['Embarked'].append(0)
								elif data_in_row[12] == 'C':
									data['Embarked'].append(1)
								elif data_in_row[12] == 'Q':
									data['Embarked'].append(2)

	if mode == 'Test':
		def average(one_item):		# Average function
			return sum(age for age in one_item)/len(one_item)

		with open(filename, 'r') as f:
			first_row = True	 # A switch
			column_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] 	# Column notes
			for column_name in column_names: 	# Create columns in dictionary
				data[column_name] = []

			missing_age = round(average(training_data['Age']), 3)
			missing_fare = round(average(training_data['Fare']), 3)

			for line in f:
				if first_row:
					first_row = False

				else:
					data_in_row = line.strip().split(',')
					for index in range(len(data_in_row)):
						if index == 1:
							data['Pclass'].append(int(data_in_row[1]))

						elif index == 4:
							if data_in_row[4] == 'male':
								data['Sex'].append(1)
							elif data_in_row[4] == 'female':
								data['Sex'].append(0)

						elif index == 5:
							if data_in_row[5] is '':
								data['Age'].append(missing_age)
							else:
								data['Age'].append(float(data_in_row[5]))

						elif index == 6:
							data['SibSp'].append(int(data_in_row[6]))

						elif index == 7:
							data['Parch'].append(int(data_in_row[7]))

						elif index == 9:
							if data_in_row[9] is '':
								data['Fare'].append(missing_fare)
							else:
								data['Fare'].append(float(data_in_row[9]))

						elif index == 11:
							if data_in_row[11] == 'S':
								data['Embarked'].append(0)
							elif data_in_row[11] == 'C':
								data['Embarked'].append(1)
							elif data_in_row[11] == 'Q':
								data['Embarked'].append(2)
	return data
	# data['Survived'] = list(map(int, data['Survived']))


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	# One hot encoding for a new category Male
	if feature == 'Sex':
		data['Sex_0'] = []		# Create a default column for female
		data['Sex_1'] = []		# Create a default column for male
		for sex in range(len(data[feature])):
			if data[feature][sex] == 0:	 # if female
				data['Sex_0'].append(1)
				data['Sex_1'].append(0)
			elif data[feature][sex] == 1:	 # if male
				data['Sex_0'].append(0)
				data['Sex_1'].append(1)

	elif feature == 'Pclass':
		data['Pclass_0'] = []
		data['Pclass_1'] = []
		data['Pclass_2'] = []
		for pclass in range(len(data[feature])):
			if data[feature][pclass] == 1:
				data['Pclass_0'].append(1)
				data['Pclass_1'].append(0)
				data['Pclass_2'].append(0)
			elif data[feature][pclass] == 2:
				data['Pclass_0'].append(0)
				data['Pclass_1'].append(1)
				data['Pclass_2'].append(0)
			elif data[feature][pclass] == 3:
				data['Pclass_0'].append(0)
				data['Pclass_1'].append(0)
				data['Pclass_2'].append(1)

	elif feature == 'Embarked':
		data['Embarked_0'] = []
		data['Embarked_1'] = []
		data['Embarked_2'] = []
		for embarked in range(len(data[feature])):
			if data[feature][embarked] == 0:
				data['Embarked_0'].append(1)
				data['Embarked_1'].append(0)
				data['Embarked_2'].append(0)
			elif data[feature][embarked] == 1:
				data['Embarked_0'].append(0)
				data['Embarked_1'].append(1)
				data['Embarked_2'].append(0)
			elif data[feature][embarked] == 2:
				data['Embarked_0'].append(0)
				data['Embarked_1'].append(0)
				data['Embarked_2'].append(1)

	data.pop(feature)
	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	nor_data = {}	 # Create a dictionary for normalized data
	for column in data:
		nor_data[column] = []	 # Create a empty list for each column
		max_ = np.max(data[column])		# Acquire the maximum
		min_ = np.min(data[column])		# Acquire the minimum
		for each_value in data[column]:		# Do the normalization
			nor_data[column].append((each_value-min_)/(max_-min_))
	data = nor_data		# Re-assign the dictionary
	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0

	# Step 2 : Start training
	for epoch in range(num_epochs):
		for data_index in range(len(labels)):

			# Step 3 : Feature extraction
			feature = {}
			if degree == 1:
				for i in range(len(keys)):
					feature[keys[i]] = inputs[keys[i]][data_index]
			elif degree == 2:
				for i in range(len(keys)):
					feature[keys[i]] = inputs[keys[i]][data_index]
				for i in range(len(keys)):
					for j in range(i, len(keys)):
						feature[keys[i] + keys[j]] = inputs[keys[i]][data_index] * inputs[keys[j]][data_index]
			y = labels[data_index]
			h = 1 / (1 + math.exp(-dotProduct(weights, feature)))

			# Step 4 : Update weights
			increment(weights, -alpha*(h-y), feature)

	return weights
