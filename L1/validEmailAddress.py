"""
File: validEmailAddress.py
Name: Cheng-Chu Chung
----------------------------
This file shows what a feature vector is
and what a weight vector is for valid email 
address classifier. You will use a given 
weight vector to classify what is the percentage
of correct classification.

Accuracy of this model: TODO: 0.6538461538461539
"""

WEIGHT = [                           # The weight vector selected by Jerry
	[0.4],                           # (see assignment handout for more details)
	[0.4],
	[0.2],
	[0.2],
	[0.9],
	[-0.65],
	[0.1],
	[0.1],
	[0.1],
	[-0.7]
]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	maybe_email_list = read_in_data()
	accuracy = 0
	index = 1
	for maybe_email in maybe_email_list:
		score = 0
		feature_vector = feature_extractor(maybe_email)
		# print(maybe_email, feature_vector)
		score += feature_vector[0] * WEIGHT[0][0]
		score += feature_vector[1] * WEIGHT[1][0]
		score += feature_vector[2] * WEIGHT[2][0]
		score += feature_vector[3] * WEIGHT[3][0]
		score += feature_vector[4] * WEIGHT[4][0]
		score += feature_vector[5] * WEIGHT[5][0]
		score += feature_vector[6] * WEIGHT[6][0]
		score += feature_vector[7] * WEIGHT[7][0]
		score += feature_vector[8] * WEIGHT[8][0]
		score += feature_vector[9] * WEIGHT[9][0]
		# print(score)
		if score > 0:
			print('Available')
			if index > 13:
				accuracy += 1
		else:
			print('Not available')
			if index <= 13:
				accuracy += 1
		index += 1
	accuracy /= len(maybe_email_list)
	print('Accuracy:', accuracy)


def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with 26 values of 0's or 1's
	"""
	feature_vector = [0] * len(WEIGHT)
	for i in range(len(feature_vector)):
		if i == 0:	 # First feature vector
			feature_vector[i] = 1 if '@' in maybe_email else 0
		elif i == 1:
			if feature_vector[0]:	 # if True or False
				feature_vector[i] = 1 if '.' not in maybe_email.split('@')[0] else 0
				# maybe_email.split('@')[0] is the first string
		elif i == 2:
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')[0]) > 0 else 0
		elif i == 3:
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')[1]) > 0 else 0
		elif i == 4:
			if feature_vector[0]:
				index = maybe_email.find('@')
				feature_vector[i] = 1 if '.' in maybe_email[index:] else 0
		elif i == 5:
			feature_vector[i] = 1 if ' ' not in maybe_email else 0
		elif i == 6:
			feature_vector[i] = 1 if maybe_email[-4:] == '.com' else 0
		elif i == 7:
			feature_vector[i] = 1 if maybe_email[-4:] == '.edu' else 0
		elif i == 8:
			feature_vector[i] = 1 if maybe_email[-3:] == '.tw' else 0
		elif i == 9:
			feature_vector[i] = 1 if len(maybe_email) > 10 else 0
		###################################
		#                                 #
		#        0.6538461538461539       #
		#                                 #
		###################################
	return feature_vector


def read_in_data():
	"""
	:return: list, containing strings that might be valid email addresses
	"""
	# TODO:
	with open(DATA_FILE, 'r', encoding="utf-8") as f:
		email_list = []
		for line in f:
			email_list.append(line.strip())
		# print(email_list)
		return email_list


if __name__ == '__main__':
	main()
