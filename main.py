import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from preprocessor import Preprocessor
from sklearn.ensemble import RandomForestClassifier



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
    preprocessor_test = Preprocessor(test_file)

    #data = preprocessor.get_matrix(['PassengerId','Pclass', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Fare', 'Survived'])
    data = preprocessor.get_matrix(['Pclass', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Fare', 'Survived','PassengerId'])
    data_test = preprocessor_test.get_matrix(['Pclass', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Fare', 'Survived','PassengerId'])

    #print(data[:10,:])


    TEST_SIZE = data_test.shape[0]
    train, train_labels_pid = data[:TEST_SIZE,:-1], data[:TEST_SIZE,-2:]
    train, train_labels, train_pid = data[:TEST_SIZE, :-1], data[:TEST_SIZE, -2], data[:TEST_SIZE, -1]
    test, test_labels = data_test[:TEST_SIZE,:-1], data_test[:TEST_SIZE,-2]
    print(train.shape)
    print(data.shape)
    print(test.shape)
    #print(train_labels_pid)
    # print(train_labels_pid)

    #print(train)

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(train, train_labels)
    print(clf.feature_importances_)
    print(test[0].reshape(1,-1))
    print("predict: ", clf.predict(test[0].reshape(1,-1)))
    print("label: ", test_labels[0])
    print("Decision Path:\n", clf.decision_path(test[0].reshape(1,-1)))

    confusion_matrix = np.zeros((2,2))
    for i in range(test_labels.shape[0]):
        x = clf.predict(test[i].reshape(1,-1))
        y = test_labels[i]
        # print(i)
        print(x)
        print(y)
        #confusion_matrix[x][y] += 1

    accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    print(accuracy)

if __name__ == '__main__':
    main()