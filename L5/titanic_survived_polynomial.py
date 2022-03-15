"""
File: titanic_survived_polynomial.py
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
TEST_DATA_PATH = 'titanic_data/test.csv'
NUM_EPOCHS = 10000
ALPHA = 0.003   # You can modify
LAMBDA = 0.00001    # Constrain loss


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
    total = 0
    for i in range(len(lst1)):
        total += lst1[i] * lst2[i]
    return total


def main():
    # Milestone 1
    training_data = data_pre_processing(TRAIN_DATA_PATH)

    # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    weights = [0]*len(training_data[0][0])

    # Milestone 2
    training(training_data, weights)

    print('weights:', weights)

    # Milestone 3
    predict(training_data, weights)

    # Milestone 4
    test(weights)


# Milestone 1
def data_pre_processing(filename, mode='train'):
    """
    Read the training data from TRAIN_DATA_PATH and get ready for training!
    :return: list[Tuple(data, label)], the value of each data on
    (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], 'Survived')
    """
    all_data = []

    with open(filename, 'r') as f:
        first = True
        for line in f:
            if first:
                first = False
            else:
                line = line.strip()
                if mode == 'train':
                    feature_vector, y = feature_extractor(line, mode)
                    all_data.append((feature_vector, y))
                else:
                    feature_vector = feature_extractor(line, mode)
                    all_data.append(feature_vector)
    return all_data


def feature_extractor(line, mode):
    """
    : param line: str, the line of data extracted from the training set
    : return: Tuple(list, label), the feature vector and the true label
    """
    line = line.strip()
    data_lst = line.split(',')
    #[Id, Surv, Pclass, F_Name, L_Name, Sex5, Age, SibSp, ParCh, Ticket, Fare10, Cabin, Embarked]
    ans = []
    if mode == 'train':
        y = int(data_lst[1])    # Surv
        start = 2
    else:
        y = -1
        start = 1
    for i in range(len(data_lst)):
        if i == start:
            # Pclass
            ans.append((int(data_lst[i])-1) / (3-1))
            ans.append(((int(data_lst[i]) - 1) / (3 - 1))**2)
        elif i == start+3:
            # Gender
            if data_lst[i] == 'male':
                ans.append(1)
            else:
                ans.append(0)
        elif i == start+4:
            # Age
            if data_lst[i].isdigit():
                ans.append((float(data_lst[i]) - 0.42) / (80-0.42))
                ans.append(((float(data_lst[i]) - 0.42) / (80 - 0.42))**2)
            else:
                ans.append((29.699 - 0.42) / (80-0.42))
                ans.append(((29.699 - 0.42) / (80 - 0.42))**2)
        elif i == start+5:
            # SibSp
            ans.append((int(data_lst[i]) - 0) / 8)
        elif i == start+6:
            # Parch
            ans.append((int(data_lst[i]) - 0) / 6)
        elif i == start+8:
            # Fare
            if data_lst[i]:
                ans.append((float(data_lst[i]) - 0) / 512.3292)
                # ans.append(((float(data_lst[i]) - 0) / 512.3292)**2)
            else:
                ans.append(32.2/512.3292)
                # ans.append((32.2 / 512.3292)**2)
        elif i == start+10:
            if data_lst[i] == 'S':
                ans.append(0 / 2)
            elif data_lst[i] == 'C':
                ans.append(1 / 2)
            elif data_lst[i] == 'Q':
                ans.append(2 / 2)
            else:
                ans.append(0 / 2)
    if mode == 'train':
        return ans, y
    return ans


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
            # w = w - alpha*dL_dw
            # w = w - alpha*(h-y)*x
            h = sigmoid(dot(weights, x))
            cost += (-(y*math.log(h)+(1-y)*math.log(1-h)) + LAMBDA/2*dot(weights, weights))
            # w = w - alpha*((h-y)*x+λ*w)
            for i in range(len(weights)):
                weights[i] = weights[i] - ALPHA*((h-y)*x[i]+LAMBDA*weights[i])
            #################################
        if epoch % 500 == 0:
            print('Cost over all data:', cost/len(training_data))


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
        prediction = get_prediction(x, weights)
        print('True Label: ' + str(y) + ' --------> Predict: ' + str(prediction))
        if y == prediction:
            acc += 1
        num_data += 1
    print('-------------------------------------')
    print('Training Acc: ' + str(acc / num_data))


def get_prediction(x, weights):
    """
    : param x: list[float], the value of each data on
    ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    : param weights: list[float], the weight vector (the parameters on each feature)
    : return: float, the score of x (if it is > 0 then the passenger may survive)
    """
    score = dot(x, weights)
    return 1 if score > 0 else 0


def test(weights):
    """
    :param weights: list[float], the weight vector of titanic features
    ------------------------------------------------------------------
    Predict the result in test.csv by extracting the features and
    getting the score from dot(weights, test_feature_vector)
    """
    test_data = data_pre_processing(TEST_DATA_PATH, mode='test')
    prediction_list = []
    for i in range(len(test_data)):
        k = dot(test_data[i], weights)
        # h = sigmoid(k)
        prediction = 1 if k > 0 else 0   # or if h > 0.5 else 0
        prediction_list.append(prediction)
    # print(prediction_list)
    out_file(prediction_list, 'titanic_my_submission_4_regularization.csv')
    ##############################
    #                            #
    #           TODO:            #
    #                            #
    ##############################


def out_file(predictions, filename):
    """
    : param predictions: numpy.array, a list-like data structure that stores 0's and 1's
    : param filename: str, the filename you would like to write the results to
    """
    print('\n===============================================')
    print(f'Writing predictions to --> {filename}')
    with open(filename, 'w') as f:
        f.write('PassengerId,Survived\n')   # No space between ,
        start_id = 892  # Start from here
        for i in range(len(predictions)):
            f.write(str(start_id+i)+','+str(predictions[i])+'\n')

    ##############################
    #                            #
    #           TODO:            #
    #                            #
    ##############################
    print('===============================================')


if __name__ == '__main__':
    main()
