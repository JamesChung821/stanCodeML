"""
File: titanic_pandas.py
Name: 
---------------------------
This file shows how to pandas and sklearn
packages to build a machine learning project
from scratch by their high order abstraction.
The steps of this project are:
1) Data pre-processing by pandas
2) Learning by sklearn
3) Test on D_test
"""

import pandas as pd
from sklearn import linear_model, preprocessing


# Constants - filenames for data set
TRAIN_FILE = 'titanic_data/train.csv'             # Training set filename
TEST_FILE = 'titanic_data/test.csv'               # Test set filename

# Global variable
nan_cache = {}                                    # Cache for test set missing data


def main():

	# Data cleaning
	train_data = data_preprocess(TRAIN_FILE, mode='Train')
	test_data = data_preprocess(TEST_FILE, mode='Test')

	# Extract true labels
	y_train = train_data.pop('Survived')
	print(y_train)
	print(train_data.count())	 # No survived

	# Abandon features ('PassengerId', 'Name', 'Ticket', 'Cabin')
	train_data.pop('PassengerId')
	train_data.pop('Name')
	train_data.pop('Ticket')
	train_data.pop('Cabin')
	print(train_data.count())
	test_data.pop('PassengerId')
	test_data.pop('Name')
	test_data.pop('Ticket')
	test_data.pop('Cabin')
	print(test_data.count())
	# Extract features ('Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked')

	# Normalization / Standardization
	# normalizer = preprocessing.MinMaxScaler() 	# Object
	# x_train = normalizer.fit_transform(train_data)
	# print(x_train)

	standardizer = preprocessing.StandardScaler()	 # Zero center is better
	x_train = standardizer.fit_transform(train_data)
	x_test = standardizer.transform(test_data)		# no fit, using standardizer to transform
	print(x_train)
	#############################
	# Degree 1 Polynomial Model #
	h = linear_model.LogisticRegression()
	classifier = h.fit(x_train, y_train)	 # fit no transform
	acc = classifier.score(x_train, y_train)
	print('Training Acc:', acc)
	#############################

	# Test dataset
	predictions = classifier.predict(x_test)
	print(predictions)
	out_file(predictions, 'pandas_standardization.csv')
	#############################
	# Degree 2 Polynomial Model #
	#############################
	poly_phi_extractor = preprocessing.PolynomialFeatures(degree=2)
	x_train = poly_phi_extractor.fit_transform(x_train)		# data preprocessing
	x_test = poly_phi_extractor.transform(x_test)
	h = linear_model.LogisticRegression(max_iter=2000)
	classifier = h.fit(x_train, y_train)	 # Training
	acc = classifier.score(x_train, y_train)
	print('Training Acc:', acc)
	predictions = classifier.predict(x_test)
	out_file(predictions, 'degree.csv')
	# Test dataset
	

def data_preprocess(filename, mode='Train'):
	"""
	: param filename: str, the csv file to be read into by pd
	: param mode: str, the indicator of training mode or testing mode
	-----------------------------------------------
	This function reads in data by pd, changing string data to float, 
	and finally tackling missing data showing as NaN on pandas
	"""

	# Read in data as a column based DataFrame
	data = pd.read_csv(filename)
	print(data.count())

	if mode == 'Train':
		# Cleaning the missing data in Age column by replacing NaN with its median
		# print(data.Age)
		# print(data['Age'].dropna())		# remove nan
		age_median = data['Age'].median()
		print('---Median---')
		print(age_median)
		print('---Before filling---')
		print(data.Age)
		data['Age'] = data['Age'].fillna(age_median)
		print('---filling a number---')
		print(data.Age)

		# Filling the missing data in Embarked column with 'S'
		data['Embarked'].fillna('S', inplace=True) 	# No need to reassign

		# Cache some data for test set (Age and Fare)
		nan_cache['Age'] = age_median		# Temporary dict to avoid stack frame
		nan_cache['Fare'] = data['Fare'].median()

	else:
		# Fill in the NaN cells by the values in nan_cache to make it consistent
		data['Fare'].fillna(nan_cache['Fare'], inplace=True)	 # Be careful to use training data to fill in testing data
		data['Age'].fillna(nan_cache['Age'], inplace=True)
		print(data.count())

	# Changing 'male' to 1, 'female' to 0
	data.loc[data.Sex == 'male', 'Sex'] = 1
	data.loc[data.Sex == 'female', 'Sex'] = 0
	# Changing 'S' to 0, 'C' to 1, 'Q' to 2
	data.loc[data.Embarked == 'S', 'Embarked'] = 0
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2
	return one_hot_encoding(data)
	

def out_file(predictions, filename):
	"""
	: param predictions: numpy.array, a list-like data structure that stores 0's and 1's
	: param filename: str, the filename you would like to write the results to
	"""
	print('\n===============================================')
	print(f'Writing predictions to --> {filename}')
	with open(filename, 'w') as out:
		out.write('PassengerId,Survived\n')
		start_id = 892
		for ans in predictions:
			out.write(str(start_id)+','+str(ans)+'\n')
			start_id += 1
	print('===============================================')


def one_hot_encoding(data):
	"""
	:param data: pd.DataFrame, the 2D data
	----------------------------------------ㄣ--------
	Extract important categorical data, making it a new one-hot vector
	"""
	# One hot encoding for a new category Male
	data['Male'] = 0	 # Create a default column
	data.loc[data.Sex == 1, 'Male'] = 1

	# One hot encoding for a new category Female
	data['Female'] = 0
	data.loc[data.Sex == 0, 'Female'] = 1
	# No need Sex anymore!
	data.pop('Sex')
	# One hot encoding for a new category FirstClass
	data['FirstClass'] = 0
	data.loc[data.Pclass == 1, 'FirstClass'] = 1
	# One hot encoding for a new category SecondClass
	data['SecondClass'] = 0
	data.loc[data.Pclass == 2, 'SecondClass'] = 1
	# One hot encoding for a new category ThirdClass
	data['ThirdClass'] = 0
	data.loc[data.Pclass == 3, 'ThirdClass'] = 1
	# No need Pclass anymore!
	data.pop('Pclass')
	print(data)

	return data


if __name__ == '__main__':
	main()
