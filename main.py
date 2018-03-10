import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from preprocessor import Preprocessor


def main():
    train_file = './data/titanic_train.csv'
    test_file = './data/titanic_test.csv'
    full_file = './data/titanic_full.csv'

    columns = [
        'PassengerId',
        'Pclass',
        'Name',
        'Sex',
        'Age',
        'SibSp',
        'Parch',
        'Ticket',
        'Fare',
        'Cabin',
        'Embarked',
        'Survived'
    ]

    preprocessor = Preprocessor(train_file)

    data = preprocessor.get_matrix(['PassengerId', 'Sex', 'Ticket', 'Pclass', 'Fare', 'Survived'])
    print(data[:10,:])
    exit()


    TEST_SIZE = 800
    train, train_labels = data[:TEST_SIZE,:-1], data[:TEST_SIZE,-1]
    test, test_labels = data[TEST_SIZE:,:-1], data[TEST_SIZE:,-1]

    print(train)


if __name__ == '__main__':
    main()