"""
File: titanic_level2.py
Name: Cheng-Chu Chung
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle. Hyperparameters are hidden by the library!
This abstraction makes it easy to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'
			 data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)	 # Read the file in a dataframe form
	# column_names = row_data.head(0).columns if you need all the column names

	if mode == 'Train':
		column_names = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
		processed_data = pd.DataFrame({})  # Create a temporary DataFrame for the processing
		for column_name in column_names:  # Assign the columns and data we need into the temporary DataFrame
			processed_data[column_name] = data[column_name]

		processed_data.dropna(inplace=True)  # Drop all the NaN and remember to put inplace to replace the row data
		labels = processed_data['Survived']  # Save labels

		# Changing 'male' to 1, 'female' to 0
		processed_data.loc[processed_data.Sex == 'male', 'Sex'] = 1
		processed_data.loc[processed_data.Sex == 'female', 'Sex'] = 0

		# Changing 'S' to 0, 'C' to 1, 'Q' to 2
		processed_data.loc[processed_data.Embarked == 'S', 'Embarked'] = 0
		processed_data.loc[processed_data.Embarked == 'C', 'Embarked'] = 1
		processed_data.loc[processed_data.Embarked == 'Q', 'Embarked'] = 2

		data = pd.DataFrame({})
		for column_name in column_names:  # Assign the processed data into the DataFrame
			if column_name is not 'Survived':
				data[column_name] = processed_data[column_name]

		return data, labels

	elif mode == 'Test':
		column_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
		processed_data = pd.DataFrame({})  # Create a temporary DataFrame for the processing
		for column_name in column_names:  # Assign the columns and data we need into the temporary DataFrame
			processed_data[column_name] = data[column_name]

		missing_age = round(training_data['Age'].mean(), 3)
		missing_fare = round(training_data['Fare'].mean(), 3)

		processed_data['Age'].fillna(missing_age, inplace=True)  # Be careful to use training data to fill in testing data
		processed_data['Fare'].fillna(missing_fare, inplace=True)

		# Changing 'male' to 1, 'female' to 0
		processed_data.loc[processed_data.Sex == 'male', 'Sex'] = 1
		processed_data.loc[processed_data.Sex == 'female', 'Sex'] = 0

		# Changing 'S' to 0, 'C' to 1, 'Q' to 2
		processed_data.loc[processed_data.Embarked == 'S', 'Embarked'] = 0
		processed_data.loc[processed_data.Embarked == 'C', 'Embarked'] = 1
		processed_data.loc[processed_data.Embarked == 'Q', 'Embarked'] = 2

		data = processed_data

		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""

	# data = pd.get_dummies(data, columns=[feature])

	# distinct_vals = data[feature].value_count()
	# for i in range(len(distinct_vals)):
	# 	data[feature + '_' + str(i)] = 0
	# 	if feature == 'Pclass':
	# 		value = i + 1
	# 	else:
	# 		value = i
	# 	data.loc[data[feature] == value, feature + '_' + str(i)] = 1
	# data.pop(feature)

	if feature == 'Sex':
		# One hot encoding for a new category Female
		data['Sex_0'] = 0  # Create a default column
		data.loc[data[feature] == 0, 'Sex_0'] = 1
		# One hot encoding for a new category Male
		data['Sex_1'] = 0
		data.loc[data[feature] == 1, 'Sex_1'] = 1

	elif feature == 'Pclass':
		# One hot encoding for a new category Pclass
		data['Pclass_0'] = 0
		data.loc[data[feature] == 1, 'Pclass_0'] = 1
		data['Pclass_1'] = 0
		data.loc[data[feature] == 2, 'Pclass_1'] = 1
		data['Pclass_2'] = 0
		data.loc[data[feature] == 3, 'Pclass_2'] = 1

	elif feature == 'Embarked':
		# One hot encoding for a new category Embarked
		data['Embarked_0'] = 0
		data.loc[data[feature] == 0, 'Embarked_0'] = 1
		data['Embarked_1'] = 0
		data.loc[data[feature] == 1, 'Embarked_1'] = 1
		data['Embarked_2'] = 0
		data.loc[data[feature] == 2, 'Embarked_2'] = 1

	# No need the feature anymore!
	data.pop(feature)

	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	standardizer = preprocessing.StandardScaler()
	data = standardizer.fit_transform(data)		# Data is well processed, so we can directly do the standardization
	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy
	on degree1; ~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimals)
	TODO: real accuracy on degree1 -> 80.19662921 %
	TODO: real accuracy on degree2 -> 83.70786517 %
	TODO: real accuracy on degree3 -> 87.64044944 %
	"""
	train_data, y_train = data_preprocess(TRAIN_FILE, mode='Train')		# Extract data and labels

	encoding_list = ['Sex', 'Pclass', 'Embarked']	 # List for one-hot encoding
	for feature in encoding_list:
		data = one_hot_encoding(train_data, feature)
		train_data = data

	standardizier = preprocessing.StandardScaler()	 # Call a standardization object
	x_train = standardizier.fit_transform(train_data)	 # Do the standardization

	for degree in [1, 2, 3]:
		poly_phi_extractor = preprocessing.PolynomialFeatures(degree=degree)	 # Call a higher order feature extractor
		high_d_x_train = poly_phi_extractor.fit_transform(x_train)	 # Do the data fitting

		h = linear_model.LogisticRegression(max_iter=10000)		# Call a model for training
		classifier = h.fit(high_d_x_train, y_train)		# Update weights for training

		acc = classifier.score(high_d_x_train, y_train) * 100
		print(f'Real accuracy on degree{degree} --> {round(acc, 8)} %')


if __name__ == '__main__':
	main()
