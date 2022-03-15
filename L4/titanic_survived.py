"""
File: titanic_survived.py
Name:
----------------------------------
This file contains 3 of the most important steps
in machine learning:
1) Data pre-processing
2) Training
3) Predicting
"""

import math

TRAIN_DATA_PATH = 'titanic_data/train.csv'
NUM_EPOCHS = 1000
ALPHA = 0.01    # Alpha can be small to solve fluctataion issue


def main():
    # Milestone 1
    training_data = data_pre_processing()   # A lot of tuple

    # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    weights = [0]*len(training_data[0][0])  # list[Tuple(data, label)]

    # Milestone 2
    training(training_data, weights)

    # Milestone 3
    predict(training_data, weights)


# Milestone 1
def data_pre_processing():
    """
    Read the training data from TRAIN_DATA_PATH and get ready for training!
    :return: list[Tuple(data, label)], the value of each data on
    (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 'Survived')
    """
    training_data = []
    with open(TRAIN_DATA_PATH, 'r') as f:
        is_first = True
        for line in f:
            if is_first:
                is_first = False
            else:
                feature_vector, label = feature_extractor(line)
                training_data.append((feature_vector, label))   # Tuple is immutable
    return training_data


def feature_extractor(line):
    """
    : param line: str, the line of data extracted from the training set
    : return: Tuple(list, label), feature_vector and label of a passenger
    """
    # line = line.strip() ##########################3
    data_list = line.split(',')
    feature_vector = []
    label = int(data_list[1])   # <---------------------------- whether is survived
    for i in range(len(data_list)):
        if i == 2:
            # Pclass
            feature_vector.append((int(data_list[i])-1)/(3-1))  # normalized
        elif i == 5:    # , after name
            # Gender
            if data_list[i] == 'male':
                feature_vector.append(1)
            else:
                feature_vector.append(0)
        elif i == 6:    # missing data or float, average is 29.699
            # Age
            if data_list[i].isdigit():
                feature_vector.append((float(data_list[i])-0.42)/(80-0.42))
            else:
                feature_vector.append((29.699-0.42)/(80-0.42))
        elif i == 7:
            # SibSp
            feature_vector.append((int(data_list[i])-0)/(8-0))
        elif i == 8:
            # Parch
            feature_vector.append((int(data_list[i])-0)/(6-0))
        elif i == 10:
            # Fare
            feature_vector.append((float(data_list[i])-0)/(512.3292))
        elif i == 12:
            # Embarked, S=0, C=1, Q=2
            if data_list[i] == 'S':
                feature_vector.append(0)
            elif data_list[i] == 'C':
                feature_vector.append(1/2)
            elif data_list[i] == 'Q':
                feature_vector.append(2/2)
            else:
                feature_vector.append(0)

    return feature_vector, label


# Milestone 2
def training(training_data, weights):
    """
    : param training_data: list[Tuple(data, label)], the value of each data on
    (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 'Survived')
    : param weights: list[float], the weight vector (the parameters on each feature)
    """
    for epoch in range(NUM_EPOCHS):
        cost = 0
        for x, y in training_data:
            #################################
            # h = sigmoid(sum(x[i]*weights[i] for i in range(len(weights))))
            h = sigmoid(dot(x, weights))
            loss = -(y*math.log(h) + (1-y)*math.log(1-h))
            # k = dot(x, weights)
            # h = sigmoid(k)
            cost += loss
            # raise Exception('Not implemented yet')

            # Gradient Descent
            for i in range(len(weights)):
                weights[i] = weights[i] - ALPHA * (h-y)*x[i]    # dL_dW
            #################################
        cost /= len(training_data)
        if epoch%100 == 0:
            print('Cost over all data:', cost)


def sigmoid(k):
    """
    :param k: float, linear function value
    :return: float, probability of the linear function value
    """
    return 1/(1+math.exp(-k))


def dot(lst1, lst2):
    """
    : param lst1: list, the feature vector
    : param lst2: list, the weights
    : return: float, the dot product of 2 list
    """
    return sum(lst1[i]*lst2[i] for i in range(len(lst1)))


# Milestone 3
def predict(training_data, weights):
    """
    : param training_data: list[Tuple(data, label)], the value of each data on
    (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 'Survived')
    : param weights: list[float], the weight vector (the parameters on each feature)
    """
    acc = 0
    num_data = 0
    for x, y in training_data:
        predict = get_prediction(x, weights)
        print('True Label: ' + str(y) + ' --------> Predict: ' + str(predict))
        if y == predict:
            acc += 1
        num_data += 1
    print('---------------------------')
    print('Acc: ' + str(acc / num_data))
    print(weights)


def get_prediction(x, weights):
    """
    : param x: list[float], the value of each data on
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    : param weights: list[float], the weight vector (the parameters on each feature)
    : return: float, the score of x (if it is > 0 then the passenger may survive)
    """
    k = dot(x, weights)
    if k > 0:
        return 1
    return 0
    # k = dot(x, weights)
    # h = sigmoid(k)
    # if h > 0.5:
    #     return 1
    # return 0


if __name__ == '__main__':
    main()
