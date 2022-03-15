"""
File: boston_housing_competition.py
Name: Cheng-Chu Chung
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientist!
--------------------------------
TODO: Assign SINGLE_LEARNER as True or False to run a specific model or test all models, respectively.
TODO: You could give different parameters for NO_CORRELEATION, OUTLINERS, RANDOM_STATE, TEST_SIZE, PCA, DEGREES.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model, metrics, model_selection, tree, decomposition, ensemble
from xgboost import XGBRegressor
import seaborn as sns
import timeit
import streamlit as st
import shap

TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'

# Confirmed outlier items = ['dis', 'crim', 'zn', 'rm', 'ptratio', 'black', 'lstat']

# Pre-step 1: Set data parameters
NO_CORRELATION = []	 # These conditions have no relationship with others 'chas', 'black'
OUTLIERS = []	 # The conditions that have some outliers which need to be removed by using IQR
RANDOM_STATE = [0]	 # Random state number
# RANDOM_STATE = list(k for k in range(43))
TEST_SIZE = [0.1]		# How much percentage of validation data you want to split from the train data

# Pre-step 2: Decide the model parameters
PCA = list(i for i in range(13-len(NO_CORRELATION), 8-len(NO_CORRELATION), -1)) 	# Principle Component Analysis
DEGREES = [1, 2]		# PolynomialFeatures
SINGLE_LEARNER = True	 # Only one learner applied?  <------------------------------- Check which way you want to run
SPECIFIC_LEARNER = [linear_model.HuberRegressor(max_iter=20000)]	 # Which learner?
# If you want to run all the learners
LEARNERS = [linear_model.LinearRegression(),
			linear_model.Ridge(max_iter=20000),
			linear_model.ElasticNet(max_iter=20000),
			linear_model.RANSACRegressor(linear_model.LinearRegression(), max_trials=20000, min_samples=50),
			linear_model.TheilSenRegressor(max_iter=20000),
			linear_model.HuberRegressor(max_iter=20000),
			linear_model.QuantileRegressor(),
			linear_model.SGDRegressor(max_iter=20000),
			linear_model.BayesianRidge(),
			tree.DecisionTreeRegressor(),
			ensemble.RandomForestRegressor(),
			XGBRegressor()]
VAL_SCORE_THRESHOLD = 2		# Out file if the score is less than this threshold.
TRAIN_SCORE_THRESHOLD = 5	 # Out file if the score is less than this threshold.
IF_OUTFILE = False		# Whether you want to output the file
IF_SHAP_ON = False
"""
Linear regression model that is robust to outliers.

The Huber Regressor optimizes the squared loss for the samples where |(y - X'w) / sigma| < epsilon,
and the absolute loss for the samples where |(y - X'w) / sigma| > epsilon, 
where w and sigma are parameters to be optimized. 
The parameter sigma makes sure that if y is scaled up or down by a certain factor, 
one does not need to rescale epsilon to achieve the same robustness. 
Note that this does not take into account the fact that the different features of X may be of different scales.
The HuberRegressor is different to Ridge because it applies a linear loss to samples that are classified as outliers. 
A sample is classified as an inlier if the absolute error of that sample is lesser than a certain threshold. 
It differs from TheilSenRegressor and RANSACRegressor,
because it does not ignore the effect of the outliers but gives a lesser weight to them.
"""


def main():
	st.write("""
	# Boston House Price Prediction App
	
	This app predicts the **Boston House Price**!
	""")
	st.write('---')

	# st.sidebar.header('Specify Input Parameters')

	start = timeit.default_timer()

	# Step 1: extract the data feature from the raw data
	train__data, label__data = data_preprocess(TRAIN_FILE, mode='Train', outlier_list=OUTLIERS)  # Extract train data and labels
	test_data, index = data_preprocess(TEST_FILE, mode='Test')		# Extract test data and index
	success = 0		# To count the number of models that output
	total = 0

	default = user_input_features(train__data)
	print(train__data.crim)
	st.header('Specified Input Parameters')
	st.write(default)
	st.write('---')

	# sns.pairplot(train__data)
	# plt.show()

	# Find outliers
	# sns.boxplot(train__data['crim'])
	# sns.boxplot(train__data['zn'])
	# sns.boxplot(train__data['indus'])
	# sns.boxplot(train__data['chas'])
	# sns.boxplot(train__data['nox'])
	# sns.boxplot(train__data['rm'])
	# sns.boxplot(train__data['age'])
	# sns.boxplot(train__data['dis'])
	# sns.boxplot(train__data['rad'])
	# sns.boxplot(train__data['tax'])
	# sns.boxplot(train__data['ptratio'])
	# sns.boxplot(train__data['black'])
	# sns.boxplot(train__data['lstat'])

	# # Scatter plot
	# fig, ax = plt.subplots(figsize=(18, 10))
	# ax.scatter(train__data['indus'], train__data['tax'])
	#
	# # x-axis label
	# ax.set_xlabel('(Proportion non-retail business acres)/(town)')
	#
	# # y-axis label
	# ax.set_ylabel('(Full-value property-tax rate)/( $10,000)')

	# Pearson Correlation Coefficient
	# pcc = np.corrcoef(train__data.values.T)
	# sns.set(font_scale=1.5)
	# sns.heatmap(pcc, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12},
	# 					  yticklabels=train__data.columns, xticklabels=train__data.columns)

	plt.show()
	
	for ts in TEST_SIZE:	 # Different validation data ratio
		for state in RANDOM_STATE:	 # Shuffle times
			train_data, val_data, train_label, val_label = model_selection.train_test_split(train__data, label__data,
																					random_state=state, test_size=ts)
			# Step 2: do the data standardization
			standardizier = preprocessing.StandardScaler()  # Call a standardization object
			x_train = standardizier.fit_transform(train_data)  # Do the standardization on train data
			x_val = standardizier.transform(val_data)		# Do the standardization on validation data
			x_test = standardizier.transform(test_data)		# Do 'transform' only on testing data

			for degree in DEGREES:
				for dimension in PCA:
					# Step 3: reduce data dimension
					pca = decomposition.PCA(n_components=dimension)  # Data compression
					x_train_pca = pca.fit_transform(x_train)
					pca_var_retain = sum(pca.explained_variance_ratio_)
					x_val_pca = pca.transform(x_val)
					x_test_pca = pca.transform(x_test)

					# Step 4: raise input variables
					poly_phi_extractor = preprocessing.PolynomialFeatures(degree=degree)  # Call a higher order feature extractor
					high_d_x_train = poly_phi_extractor.fit_transform(x_train_pca)  # Do the data fitting
					high_d_x_val = poly_phi_extractor.transform(x_val_pca)
					high_d_x_test = poly_phi_extractor.transform(x_test_pca)

					# Step 5: train the data
					if SINGLE_LEARNER:
						learners = SPECIFIC_LEARNER
					else:
						learners = LEARNERS
					for learner in learners:
						# print(f'--------------------Degree{degree}--------------------')
						# print(f'--------------------{learner}')
						# print(f'No correlation: {NO_CORRELATION}')
						# print(f'Outliers: {OUTLIERS}')
						# print(f'PCA: {dimension} dimensions')
						# print(f'Validation data: {ts * 100}%')
						h = learner		# Call an object
						h.fit(high_d_x_train, train_label)  # Update weights for training

						# Step 6: view the results  --> accuracy
						acc_train = h.score(high_d_x_train, train_label) * 100
						# print(f'Real accuracy of training data on degree{degree} --> {round(acc_train, 8)} %')
						acc_val = h.score(high_d_x_val, val_label) * 100
						# print(f'Real accuracy of validation data on degree{degree} --> {round(acc_val, 8)} %')

						# Step 7: validate the prediction
						train_prediction = h.predict(high_d_x_train)
						val_prediction = h.predict(high_d_x_val)

						# Step 8: score the prediction (RMSE)
						train_score = round(metrics.mean_squared_error(train_prediction, train_label)**0.5, 5)
						# print(f'Training Score: {train_score}')
						val_score = round(metrics.mean_squared_error(val_prediction, val_label)**0.5, 5)
						# print(f'Validation Score: {val_score}')

						# Step 9: implement the model on the test data
						test_prediction = h.predict(high_d_x_test)
						# print(f'Prediction{degree}: {np.around(test_prediction, 2)}')



						total += 1

						if val_score < VAL_SCORE_THRESHOLD and train_score < TRAIN_SCORE_THRESHOLD:
							success += 1

							st.header(f'Prediction of medv\n'
									  f'st{state}_ts{ts}_pca{dimension}_degree{degree}_vs{round(val_score, 2)}')
							st.write(test_prediction)
							st.write('---')
							map_data = pd.DataFrame(
								np.random.randn(100, 2) / [50, 50] + [40.9, -73.1],
								columns=['lat', 'lon'])
							st.map(map_data, zoom=10)	 # 40.89320673626953, -73.11882380033349, 22.7, 120.3

							print(f'--------------------Degree{degree}--------------------')
							print(f'--------------------{learner}')
							print(f'No correlation: {NO_CORRELATION}')
							print(f'Outliers: {OUTLIERS}')
							print(f'Validation data: {ts * 100}%')
							print(f'PCA: {dimension} dimensions')
							print(f'Var Retain: {round(pca_var_retain*100, 2)} %')
							print(f'Real accuracy of training data on degree{degree} --> {round(acc_train, 8)} %')
							print(f'Real accuracy of validation data on degree{degree} --> {round(acc_val, 8)} %')
							print(f'Training Score: {train_score}')
							print(f'Validation Score: {val_score}')

							if IF_SHAP_ON:
								explainer = shap.TreeExplainer(h)
								shap_values = explainer.shap_values(high_d_x_train)

								st.header('Feature Importance')
								plt.title('Feature importance based on SHAP values')
								shap.summary_plot(shap_values, high_d_x_train)
								st.pyplot(bbox_inches='tight')
								st.write('---')

								plt.title('Feature importance based on SHAP values (Bar)')
								shap.summary_plot(shap_values, high_d_x_train, plot_type="bar")
								st.pyplot(bbox_inches='tight')

							if IF_OUTFILE:
								out_file(index, test_prediction,
										 f'bs_st{state}_ts{ts}_pca{dimension}_degree{degree}_vs{round(val_score, 2)}.csv')
	print('-----------Data counts')
	print(train__data.count())
	print(f'-----------------------------------------> {success}/{total} models succeed!')
	end = timeit.default_timer()
	print('Running time: %s seconds' % (end-start))
	# print(f'No correlation: {NO_CORRELATION}')
	# print(f'Outliers: {OUTLIERS}')


def data_preprocess(filename, mode='Train', outlier_list=OUTLIERS):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'
			 Tuple(data, index), if the mode is 'Test'
	"""
	data = pd.read_csv(filename)	 # Read the file in a dataframe form
	# column_names = row_data.head(0).columns if you need all the column names

	def remove_outliers(row_data, outliers):
		"""
		:param row_data: DataFrame, a 2D data structure that looks like an excel worksheet
		:param outliers: List, the columns containing outliers that you want to remove
		:return: DataFrame, a 2D data structure dropping the outliers
		"""
		column_names = ['ID', 'crim', 'zn', 'indus', 'chas',
						'nox', 'rm', 'age', 'dis', 'rad',
						'tax', 'ptratio', 'black', 'lstat']

		# Interquartile Range (IQR)
		outliers_array = np.array([])		# An empty array to store outliers index
		for item in outliers:
			q1_25percentage = np.percentile(row_data[item], 25, interpolation='midpoint')
			q3_75percentage = np.percentile(row_data[item], 75, interpolation='midpoint')
			iqr = q3_75percentage - q1_25percentage

			# Find the upper outliers
			for upper_index in np.where(row_data[item] >= (q3_75percentage + 1.5*iqr))[0]:
				if upper_index not in outliers_array:
					outliers_array = np.append(outliers_array, upper_index)

			# Find the lower outliers
			for lower_index in np.where(row_data[item] <= (q1_25percentage - 1.5*iqr))[0]:
				if lower_index not in outliers_array:
					outliers_array = np.append(outliers_array, lower_index)

		row_data.drop(outliers_array, inplace=True)		# Drop all the outliers
		return row_data

	def remove_no_correlation():
		if len(NO_CORRELATION) > 0:
			for column in NO_CORRELATION:
				data.pop(column)

	if mode == 'Train':
		column_names = ['ID', 'crim', 'zn', 'indus', 'chas',
						'nox', 'rm', 'age', 'dis', 'rad',
						'tax', 'ptratio', 'black', 'lstat', 'medv']
		remove_no_correlation()
		data = remove_outliers(data, outlier_list)		# Remove the outliers

		labels = data['medv']  # Save labels
		data.pop('medv')
		data.pop('ID')

		return data, labels

	elif mode == 'Test':
		column_names = ['ID', 'crim', 'zn', 'indus', 'chas',
						'nox', 'rm', 'age', 'dis', 'rad',
						'tax', 'ptratio', 'black', 'lstat']
		index = data['ID']
		remove_no_correlation()
		data.pop('ID')

		return data, index


def out_file(index, predictions, filename):
	"""
	: param index: numpy.array, a list-like data structure that stores index
	: param predictions: numpy.array, a list-like data structure that stores predicted price
	: param filename: str, the filename you would like to write the results to
	"""
	print('=================================================================================')
	print(f'Writing predictions to --> {filename}')
	with open(filename, 'w') as out:
		out.write('ID,medv\n')
		for i in range(len(index)):
			out.write(str(index[i])+','+str(predictions[i])+'\n')
	print('=================================================================================')
	print(' ')


def user_input_features(x):
	crim = st.sidebar.slider('crim', float(x.crim.min()), float(x.crim.max()), float(x.crim.mean()))
	zn = st.sidebar.slider('zn', float(x.zn.min()), float(x.zn.max()), float(x.zn.mean()))
	indus = st.sidebar.slider('indus', float(x.indus.min()), float(x.indus.max()), float(x.indus.mean()))
	chas = st.sidebar.slider('chas', float(x.chas.min()), float(x.chas.max()), float(x.chas.mean()))
	nox = st.sidebar.slider('nox', float(x.nox.min()), float(x.nox.max()), float(x.nox.mean()))
	rm = st.sidebar.slider('rm', float(x.rm.min()), float(x.rm.max()), float(x.rm.mean()))
	age = st.sidebar.slider('age', float(x.age.min()), float(x.age.max()), float(x.age.mean()))
	dis = st.sidebar.slider('dis', float(x.dis.min()), float(x.dis.max()), float(x.dis.mean()))
	rad = st.sidebar.slider('rad', float(x.rad.min()), float(x.rad.max()), float(x.rad.mean()))
	tax = st.sidebar.slider('tax', float(x.tax.min()), float(x.tax.max()), float(x.tax.mean()))
	ptratio = st.sidebar.slider('ptratio', float(x.ptratio.min()), float(x.ptratio.max()), float(x.ptratio.mean()))
	black = st.sidebar.slider('black', float(x.black.min()), float(x.black.max()), float(x.black.mean()))
	lstat = st.sidebar.slider('lstat', float(x.lstat.min()), float(x.lstat.max()), float(x.lstat.mean()))
	data = {'crim': crim,
			'zn': zn,
			'indus': indus,
			'chas': chas,
			'nox': nox,
			'rm': rm,
			'age': age,
			'dis': dis,
			'rad': rad,
			'tax': tax,
			'ptratio': ptratio,
			'black': black,
			'lstat': lstat}
	features = pd.DataFrame(data, index=[0])
	return features


if __name__ == '__main__':
	main()
