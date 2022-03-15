"""
File: validEmailAddress_2.py
Name: Cheng-Chu Chung
----------------------------
Please construct your own feature vectors
and try to surpass the accuracy achieved by
Jerry's feature vector in validEmailAddress.py.
feature1:  TODO: Check if only one '@' is in email address
feature2:  TODO: Check if there are special characters in " "
feature3:  TODO: Check if some strings before '@'
feature4:  TODO: Check if some strings after '@'
feature5:  TODO: Check if '.' is in strings after '@'
feature6:  TODO: Check if string length before '@' is longer than 64
feature7:  TODO: Check if the last string ends in .com/.edu/.tw
feature8:  TODO: Check if the first or last character before '@' is special character
feature9:  TODO: Check if '..'/'"'/'”' is in email address
feature10: TODO: Check if length of the email address is larger than 10

Accuracy of your model: TODO: 100.0%
"""
import numpy as np
WEIGHT = [                           # The weight vector selected by you
	[0.3],                              # (Please fill in your own weights)
	[2.5],
	[0.1],
	[0.3],
	[0.5],
	[-1.0],
	[0.1],
	[-1.0],
	[-1.5],
	[-0.6]
]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	maybe_email_list = read_in_data()
	accuracy = 0
	index = 1
	for maybe_email in maybe_email_list:
		score = 0		# Initialize the score
		weight_vector = np.array(WEIGHT).T		# Transpose the array
		feature_vector = np.array(feature_extractor(maybe_email))
		score += weight_vector.dot(feature_vector)
		# print(score)
		if score > 0:
			print(f'{index}: Available')
			if index > 13:
				accuracy += 1
		else:
			print(f'{index}: Not available')
			if index <= 13:
				accuracy += 1
		index += 1
		print('Score:', score)
	accuracy /= len(maybe_email_list)
	print(f'Accuracy: {accuracy*100}%')


def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with value 0's and 1's
	"""
	feature_vector = [0] * len(WEIGHT)
	for i in range(len(feature_vector)):
		if i == 0:		# First feature vector, check if only one '@' is in email address
			index = maybe_email.find('@')
			feature_vector[i] = 1 if '@' in maybe_email \
				and '@' not in maybe_email[index+1:] \
				else 0
		elif i == 1:	 # Check if there are special characters in " "
			if '"' in maybe_email or '”' in maybe_email:	 # Two different quotation marks identified by the index
				first_index = maybe_email.find('"')
				second_index = maybe_email.find('"', first_index+1)
				third_index = maybe_email.find('”')
				fourth_index = maybe_email.find('”', first_index + 1)
				feature_vector[i] = 1 if maybe_email[first_index+1:second_index].isalpha() is False \
					and second_index is not -1 \
					or maybe_email[third_index+1:fourth_index].isalpha() is False \
					and fourth_index is not -1 \
					else 0
		elif i == 2:	 # Check if some strings before '@'
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')[0]) > 0 else 0
				# maybe_email.split('@')[0] is the first string
		elif i == 3:	 # Check if some strings after '@'
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')[1]) > 0 else 0
		# maybe_email.split('@')[0] is the second string
		elif i == 4:	 # Check if '.' is in strings after '@'
			if feature_vector[0]:
				index = maybe_email.find('@')
				feature_vector[i] = 1 if '.' in maybe_email[index:] else 0
		elif i == 5:	 # Check if string length before '@' is longer than 64
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')[0]) > 64 else 0
		elif i == 6:	 # Check if the last string ends in .com/.edu/.tw
			feature_vector[i] = 1 if maybe_email[-4:] == '.com' \
								or maybe_email[-4:] == '.edu' \
								or maybe_email[-3:] == '.tw' \
								else 0
		elif i == 7:	 # Check if the first or last character before '@' is special character
			if feature_vector[0]:  # if only one @
				index = maybe_email.find('@')
				feature_vector[i] = 1 if maybe_email[0].isalpha() is False \
					and maybe_email[0].isnumeric() is False \
					or maybe_email[index-1] is '.' \
					else 0
		elif i == 8:	 # Check if '..'/'"'/'”' is in email address
			feature_vector[i] = 1 if '..' in maybe_email \
									or '"' in maybe_email \
									or '”' in maybe_email \
									else 0
		elif i == 9:	 # Check if length of the email address is larger than 10
			feature_vector[i] = 1 if len(maybe_email) > 10 else 0
	###################################
	#                                 #
	#               TODO:             #
	#                                 #
	###################################
	print('------------------------------------------')
	print('Email address, feature vector')
	print(maybe_email, feature_vector)
	return feature_vector


def read_in_data():
	"""
	:return: list, containing strings that may be valid email addresses
	"""
	with open(DATA_FILE, 'r', encoding="utf-8") as f:
		email_list = []
		for line in f:
			email_list.append(line.strip())
		return email_list


if __name__ == '__main__':
	main()
